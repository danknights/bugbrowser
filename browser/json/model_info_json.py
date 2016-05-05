import json
import glob
from os.path import isdir
output = {}
models = glob.glob("*")
models = filter(isdir, models)
topics, alphas, betas = zip(*[model.split("_") for model in models])
output['betas'] = sorted(set(betas))
output['alphas'] = sorted(set(alphas))
output['topics'] = sorted(set(topics))
with open('model_info.json', 'w') as outf:
    outf.write(json.dumps(output))
    
