import copy
import json
from types import SimpleNamespace
import numpy as np
import pytest
from skill3d.infra.m5_identity import content_sha256
from skill3d.segmentation.intern_decision_client import InternDecisionClient
from skill3d.segmentation.intern_decision_protocol import QUESTIONS, visual_request, fixed_relation
from skill3d.segmentation.mask_semantic_review import observe_candidate


def response_fixture(images):
    identity = {'model':'internlm/Intern-Decision-4B','backend':'native-hf-model-package',
        'dtype':'bfloat16','model_revision':'a'*40,'files':{'inference.py':{}},
        'implementation_sha256':'b'*64,'calibration':{'method':'temperature-scaling','temperature':1.99241824}}
    values = {'observed_object':'chair','visibility':'clear','mask_purity':'clean','mask_completeness':'whole'}
    answers = {}
    for field,q in QUESTIONS.items():
        probs = dict.fromkeys(q['criteria'],0.0);probs[values[field]]=1.0
        answers[field]={'type':'choice','choice':values[field],'decision':values[field],
                        'confidence':1.0,'probabilities':probs}
    return identity, {'identity_sha256':content_sha256(identity),
        'request_sha256':content_sha256(visual_request([r['url'] for r in images])),
        'result':{'model':'Intern-Decision-4B','backend':'hf','calibration':identity['calibration'],'answers':answers},
        'vision_evidence':{'image_count':3,'image_token_count':24,'pixel_values_shape':[96,768],
                           'image_grid_thw':[[1,4,4]]*3}}

@pytest.mark.parametrize('mutation',['id','identity','nan','sum','missing','decision','vision','calibration'])
def test_malformed_native_response_cannot_be_observed(mutation):
    images=[{'url':'data:image/jpeg;base64,AA=='}]*3
    identity,body=response_fixture(images)
    if mutation=='id':body['request_sha256']='wrong'
    if mutation=='identity':body['identity_sha256']='wrong'
    a=body['result']['answers']['visibility']
    if mutation=='nan':a['probabilities']['clear']=float('nan')
    if mutation=='sum':a['probabilities']['clear']=.5
    if mutation=='missing':body['result']['answers'].pop('mask_purity')
    if mutation=='decision':a['decision']='partial'
    if mutation=='vision':body['vision_evidence']['image_token_count']=0
    if mutation=='calibration':body['result']['calibration']={'temperature':1}
    client=InternDecisionClient('http://127.0.0.1:18445');client._identity=identity
    client._call=lambda *_:body
    with pytest.raises((ValueError,KeyError)):client.observe(images)


def test_visual_evidence_ignores_target_names_and_relation_preserves_materials():
    sent=[]; client=InternDecisionClient('http://127.0.0.1:18445')
    def call(path,body=None):
        if path=='/health':
            identity,_=response_fixture([{'url':'x'}]*3)
            return {'identity':identity,'identity_sha256':content_sha256(identity)}
        request=body['request'];sent.append(copy.deepcopy(request))
        identity,result=response_fixture([{'url':x} for x in request['images']]);return result
    client._call=call
    frame=np.zeros((20,24,3),np.uint8);mask=np.ones((20,24),bool)
    for name in ['paper bag','secret detector label']:
        row=observe_candidate({'candidate_id':name,'category':name,'bbox':[1,1,18,18]},frame,mask,
            client,seed=137,min_pixels=1,max_pixels=1000)
        assert row['status']=='observed'
        assert 'alternate_object' not in row['observation']
    assert sent[0]==sent[1]
    assert set(sent[0])=={'state','questions','images'}
    assert fixed_relation('radiator','heater')=='match'
    for observed in ['computer monitor','cardboard box','plastic bag']:
        assert fixed_relation(observed,'computer tower' if observed=='computer monitor' else 'paper bag')=='mismatch'
    assert fixed_relation('unknown','chair')=='uncertain'


def test_transport_failure_is_fail_closed():
    client=InternDecisionClient('http://127.0.0.1:18445')
    client._call=lambda *_: (_ for _ in ()).throw(TimeoutError('bounded failure'))
    row=observe_candidate({'candidate_id':'safe','bbox':[1,1,8,8]},np.zeros((10,10,3),np.uint8),
        np.ones((10,10),bool),client,seed=1,min_pixels=1,max_pixels=100)
    assert row['status']=='unavailable' and 'observation' not in row
