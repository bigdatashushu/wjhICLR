#!/usr/bin/env python3
"""Staged real binary experiment. Threshold searches never issue model requests."""
from __future__ import annotations
import argparse
import base64
import json
import time
import urllib.error
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from skill3d.infra.m5_identity import content_sha256, file_identity
from skill3d.segmentation.intern_binary_review import (
    InternBinaryClient, binary_request, PROTOCOL, PROTOCOL_SHA256, PROMPTS,
    threshold_search,
)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False))
    temp.replace(path)


def read(path):
    return json.loads(path.read_text())


def rows_for(args, group):
    return [r for r in read(args.manifest)['candidates'] if r['split']==group]


def run_group(args, group, version, repeat):
    output=args.output/group/version/f'r{repeat}'
    client=InternBinaryClient(args.endpoint,prompt_version=version,timeout_s=90,
                             expected_identity_sha256=args.identity)
    identity=client.identity()
    identity_path = args.output/'service_identity.json'
    if identity_path.exists() and read(identity_path) != identity:
        raise ValueError('model identity changed during experiment')
    if not identity_path.exists():
        save(identity_path,identity)
    rows=rows_for(args,group)
    for number,row in enumerate(rows,1):
        path=output/f'{row["index"]:03}.json'
        if path.exists():
            previous=read(path)
            if (previous.get('status')=='reviewed'
                    and previous.get('candidate_id')==row['candidate_id']
                    and previous.get('protocol_sha256')==PROTOCOL_SHA256
                    and previous.get('image_sha256')==row['image_sha256']
                    and previous.get('model_identity_sha256')==content_sha256(identity)):
                continue
            if previous.get('status') == 'unavailable':
                archive = path.with_name(path.stem + '.failure-' + str(time.time_ns()) + '.json')
                path.rename(archive)
            else:
                raise RuntimeError(f'Existing incompatible result: {path}')
        images=[]
        for name,expected in zip(['context','crop','mask_overlay'],row['image_sha256']):
            p=args.assets/str(row['index'])/(name+'.jpg')
            actual=file_identity(p)
            if actual['sha256']!=expected:
                raise ValueError('frozen image changed')
            images.append({'url':'data:image/jpeg;base64,'+base64.b64encode(p.read_bytes()).decode(),
                           'identity':actual})
        request=binary_request([r['url'] for r in images],row['target'],version)
        save(output/f'{row["index"]:03}.request.json',request)
        start=time.perf_counter()
        try:
            attempts=[]
            for attempt in range(3):
                try:
                    binary,response,sha=client.review_binary(images,row['target'])
                    break
                except urllib.error.HTTPError as exc:
                    attempts.append({'attempt':attempt+1,'http_status':exc.code,
                                     'error':'HTTP '+str(exc.code)})
                    if exc.code != 429 or attempt == 2:
                        raise
                    time.sleep(.25 * (attempt+1))
            result={'status':'reviewed','index':row['index'],'candidate_id':row['candidate_id'],
                'split':group,'repeat':repeat,'protocol_sha256':PROTOCOL_SHA256,
                'image_sha256':row['image_sha256'],'model_identity_sha256':content_sha256(identity),
                'binary':binary,'response':response,'request_sha256':sha,'elapsed_s':time.perf_counter()-start,
                'transport_retry_errors':attempts}
            save(path,result)
        except Exception as exc:
            save(path,{'status':'unavailable','index':row['index'],'error':f'{type(exc).__name__}: {exc}'})
            raise
        # Keep new probabilities out of progress logs until initial image review is sealed.
        print(f'{group} {version} r{repeat}: {number}/{len(rows)} completed',flush=True)
        time.sleep(.05)


def samples(args,version,repeats):
    result=[]
    for row in rows_for(args,'development'):
        for repeat in repeats:
            path=args.output/'development'/version/f'r{repeat}'/f'{row["index"]:03}.json'
            value=read(path)
            if value['status']!='reviewed':raise ValueError('incomplete development run')
            result.append({'id':row['candidate_id'],'label':row['manual'],'p_yes':value['binary']['p_yes']})
    return result


def require_review(args):
    if not args.pre_review or not args.pre_review.is_file():
        raise ValueError('Initial case-by-case image review must be sealed before selection')
    review=read(args.pre_review)
    expected={r['candidate_id'] for r in read(args.manifest)['candidates']}
    if {r['candidate_id'] for r in review['cases']}!=expected:
        raise ValueError('Initial visual review must cover every candidate')
    return file_identity(args.pre_review)


def select(args):
    review_identity=require_review(args)
    searches={v:threshold_search(samples(args,v,[1])) for v in PROMPTS}
    eligible=[v for v in PROMPTS if searches[v]['useful']]
    if not eligible:
        save(args.output/'selection.json',{'status':'no_useful_threshold','searches':searches,
            'pre_review':review_identity,'manifest':file_identity(args.manifest)})
        return
    chosen=min(eligible,key=lambda v:(-searches[v]['positive_retained'],len(PROMPTS[v]),v))
    save(args.output/'selection.json',{'status':'selected_for_repeat','prompt_version':chosen,
        'searches':searches,'protocol_sha256':PROTOCOL_SHA256,'pre_review':review_identity,
        'manifest':file_identity(args.manifest),'rule':'zero development false positives; retain most positives; shorter prompt breaks ties'})
    print('Development prompt selected:',chosen,flush=True)


def freeze(args):
    selection=read(args.output/'selection.json')
    version=selection['prompt_version']
    search=threshold_search(samples(args,version,[1,2]))
    path=args.output/'frozen_operating_point.json'
    body={'status':'frozen' if search['useful'] else 'no_useful_threshold','prompt_version':version,
          'threshold':search['threshold'],'search':search,'protocol_sha256':PROTOCOL_SHA256,
          'manifest':file_identity(args.manifest),'pre_review':require_review(args),
          'model_identity_sha256':content_sha256(read(args.output/'service_identity.json')),
          'selection_sha256':content_sha256(selection),'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
          'rule':'negative max and positive min over two development repeats; bisection; no validation labels used'}
    if path.exists():
        previous = read(path)
        if {k:v for k,v in previous.items() if k != 'created_utc'} != {k:v for k,v in body.items() if k != 'created_utc'}:
            raise ValueError('Operating point already frozen')
        body = previous
    else:
        save(path,body)
    print('Operating point:',version,search['threshold'],body['status'],flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('stage',choices=['develop','select','repeat-dev','freeze','validate'])
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--assets',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--pre-review',type=Path)
    p.add_argument('--endpoint',default='http://127.0.0.1:18445')
    p.add_argument('--identity',default='')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    contract=read(args.manifest)
    rows=contract.get('candidates',[])
    if len(rows)!=210 or len({r['candidate_id'] for r in rows})!=210:
        raise ValueError('experiment requires 210 unique frozen candidates')
    if any(r.get('split') != ('development' if r.get('qa_id') in {'2389','4566'} else 'validation')
           for r in rows):
        raise ValueError('scene split differs from frozen experiment')
    if sum(r['split']=='development' for r in rows)!=112 or sum(r['split']=='validation' for r in rows)!=98:
        raise ValueError('frozen scene split count mismatch')
    if contract['protocol_sha256']!=PROTOCOL_SHA256:raise ValueError('protocol differs from frozen manifest')
    if args.stage=='develop':
        save(args.output/'protocol.json',PROTOCOL)
        for version in PROMPTS:run_group(args,'development',version,1)
    elif args.stage=='select':select(args)
    elif args.stage=='repeat-dev':run_group(args,'development',read(args.output/'selection.json')['prompt_version'],2)
    elif args.stage=='freeze':freeze(args)
    else:
        frozen=read(args.output/'frozen_operating_point.json')
        if frozen['status']!='frozen' or frozen['manifest']!=file_identity(args.manifest):
            raise ValueError('no valid frozen operating point')
        if content_sha256(InternBinaryClient(args.endpoint).identity())!=frozen['model_identity_sha256']:
            raise ValueError('model identity changed before validation')
        for repeat in [1,2]:run_group(args,'validation',frozen['prompt_version'],repeat)

if __name__=='__main__':main()
