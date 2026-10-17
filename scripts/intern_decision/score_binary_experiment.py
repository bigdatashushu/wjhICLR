#!/usr/bin/env python3
"""Offline metrics for frozen binary prompts and independent old quality receipts."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from skill3d.segmentation.intern_binary_review import threshold_search,quality_passes


def read(p):return json.loads(p.read_text())


def quantile(values,p):
    if not values:return None
    values=sorted(values);pos=(len(values)-1)*p;lo=math.floor(pos);hi=math.ceil(pos)
    return values[lo]*(hi-pos)+values[hi]*(pos-lo) if lo!=hi else values[lo]


def classify(rows,threshold):
    count={k:0 for k in ['n','positive','negative','uncertain','semantic_tp','semantic_fp',
        'semantic_uncertain','final_tp','final_fp','final_uncertain','quality_blocked_positive',
        'failure','missing']}
    by_category={}
    for r in rows:
        count['n']+=1
        key={'true':'positive','false':'negative','uncertain':'uncertain'}[r['manual']]
        count[key]+=1
        if r.get('p_yes') is None:count['missing']+=1;continue
        passed=r['p_yes']>=threshold
        c=by_category.setdefault(r['target'],{'positive':0,'negative':0,'uncertain':0,
                                            'semantic_tp':0,'semantic_fp':0,'final_tp':0,'final_fp':0})
        c[key]+=1
        if passed:
            metric={'true':'semantic_tp','false':'semantic_fp','uncertain':'semantic_uncertain'}[r['manual']]
            count[metric]+=1
            if metric in c:c[metric]+=1
            if r['quality_pass']:
                metric={'true':'final_tp','false':'final_fp','uncertain':'final_uncertain'}[r['manual']]
                count[metric]+=1
                if metric in c:c[metric]+=1
            elif r['manual']=='true':count['quality_blocked_positive']+=1
    return {'counts':count,'by_category':by_category,'threshold':threshold}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--results',type=Path,required=True)
    p.add_argument('--pre-review',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();manifest=read(a.manifest);review={r['index']:r for r in read(a.pre_review)['cases']}
    operating=a.results/'frozen_operating_point.json';frozen=read(operating) if operating.exists() else None
    selection=a.results/'selection.json';selection=read(selection) if selection.exists() else None
    cases=[];available={};latencies=[]
    for r in manifest['candidates']:
        row={**r,'quality_pass':quality_passes(r['old_observation']),
             'review':review[r['index']],'runs':{}}
        for version in ['P0','P1','P2']:
            runs=[]
            for repeat in [1,2]:
                path=a.results/r['split']/version/f'r{repeat}'/f'{r["index"]:03}.json'
                if not path.exists():continue
                response=read(path)
                if response['status']!='reviewed':
                    runs.append({'repeat':repeat,'failure':response.get('error','unavailable')});continue
                runs.append({'repeat':repeat,**response['binary'],
                    'elapsed_s':response['elapsed_s'],'request_sha256':response['request_sha256'],
                    'raw_response_path':str(path),'model_identity_sha256':response['model_identity_sha256'],
                    'transport_retry_errors':response.get('transport_retry_errors',[])})
                available.setdefault((r['split'],version,repeat),[]).append({**row,'p_yes':response['binary']['p_yes']})
                latencies.append(response['elapsed_s'])
            row['runs'][version]=runs
        cases.append(row)
    metrics={};searches={}
    for (split,version,repeat),rows in available.items():
        name=f'{split}_{version}_r{repeat}'
        if split=='development':
            samples=[{'id':r['candidate_id'],'label':r['manual'],'p_yes':r['p_yes']} for r in rows]
            if len(rows)==112:searches[name]=threshold_search(samples)
        t=frozen['threshold'] if frozen and frozen['status']=='frozen' else .5
        metrics[name]=classify(rows,t)
        metrics[name]['expected']=112 if split=='development' else 98
    stable=[];selected=frozen['prompt_version'] if frozen else selection.get('prompt_version') if selection else None
    if selected:
        for row in cases:
            valid=[r for r in row['runs'][selected] if 'p_yes' in r]
            if len(valid)==2:
                stable.append({**row,'p_yes':min(r['p_yes'] for r in valid) if row['manual']=='true' else max(r['p_yes'] for r in valid)})
    if frozen and frozen['status']=='frozen':
        for split in ['development','validation']:
            rows=[r for r in stable if r['split']==split]
            metrics[split+'_pessimistic']=classify(rows,frozen['threshold'])
            metrics[split+'_pessimistic']['missing']=(112 if split=='development' else 98)-len(rows)
        metrics['all_pessimistic']=classify(stable,frozen['threshold'])
        t=frozen['threshold']
        metrics['neighbors']=[classify(stable,v) for v in sorted(set([max(.5,round(t-.01,2)),t,min(.99,round(t+.01,2))]))]
        disputed=[r for r in stable if not r['review'].get('historical_label_dispute')]
        metrics['disputed_labels_excluded']=classify(disputed,t)
        metrics['repeat']={'n':len(stable),'missing':210-len(stable),
           'semantic_agreement':sum((r['runs'][selected][0]['p_yes']>=t)==(r['runs'][selected][1]['p_yes']>=t) for r in stable),
           'max_probability_drift':max([abs(r['runs'][selected][0]['p_yes']-r['runs'][selected][1]['p_yes']) for r in stable],default=None)}
    if frozen and frozen['status']=='frozen':
        t=frozen['threshold']
        missed=[r for r in stable if r['manual']=='true' and not r['old_admitted']]
        metrics['old_40_missed']={
            'n':len(missed),'missing':40-len(missed),
            'semantic_recovered':sum(r['p_yes']>=t for r in missed),
            'final_recovered':sum(r['p_yes']>=t and r['quality_pass'] for r in missed),
            'still_quality_blocked':sum(r['p_yes']>=t and not r['quality_pass'] for r in missed),
            'still_semantic_rejected':sum(r['p_yes']<t for r in missed),
            'indices':{name:[r['index'] for r in missed if predicate(r)] for name,predicate in [
                ('final_recovered',lambda r:r['p_yes']>=t and r['quality_pass']),
                ('quality_blocked',lambda r:r['p_yes']>=t and not r['quality_pass']),
                ('semantic_rejected',lambda r:r['p_yes']<t)]}}
        metrics['old_admitted_lost']={'indices':[r['index'] for r in stable
            if r['manual']=='true' and r['old_admitted'] and not (r['p_yes']>=t and r['quality_pass'])]}
        metrics['official_decision_vs_threshold']={'ties':sum(
            r['p_yes']==.5 for r in stable), 'note':'official tie may choose no; configured >= threshold remains explicit'}
    archived_errors=[]
    for path in a.results.rglob('*.failure-*.json'):
        failure=read(path)
        archived_errors.append({'path':str(path),'index':failure.get('index'),'error':failure.get('error')})
    summary={'transport_errors_archived':archived_errors,'operating_point':frozen,'selection':selection,'metrics':metrics,'initial_searches':searches,
       'latency':{'requests':len(latencies),'sum_s':sum(latencies),'p50_s':quantile(latencies,.5),
                  'p95_s':quantile(latencies,.95),'throughput_per_s':len(latencies)/sum(latencies) if latencies else None},
       'quality_reference':'old target-blind model fields; no independent human quality truth',
       'reference':'historical single-review manual semantic labels; Codex notes separate',
       'cases':cases}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k not in ['cases','initial_searches','selection']},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
