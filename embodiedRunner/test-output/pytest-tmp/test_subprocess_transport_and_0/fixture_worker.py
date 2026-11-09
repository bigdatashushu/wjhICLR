import json,sys
for line in sys.stdin:
 c=json.loads(line)
 print(json.dumps({'id':c['id'],'version':'embodiedrunner-navigation/1','ok':True,'data':{'echo':c['op']}}),flush=True)
