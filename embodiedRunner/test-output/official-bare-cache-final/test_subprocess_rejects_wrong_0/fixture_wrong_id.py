import json,sys
for line in sys.stdin:
 print(json.dumps({'id':999,'version':'embodiedrunner-navigation/1','ok':True,'data':{}}),flush=True)
