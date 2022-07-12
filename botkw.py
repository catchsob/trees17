import os
import argparse
from glob import glob


def rep(filename, secret=None, token=None, model=None, labels=None, modelin=None, modelout=None):
    if not secret and not token and not model and not labels:
        return False
    
    m = {'YOUR_CHANNEL_ACCESS_TOKEN': token,
         'YOUR_CHANNEL_SECRET': secret,
         'YOUR_LABELS': labels,
         'YOUR_MODELIN': modelin,
         'YOUR_MODELOUT': modelout,
         'YOUR_MODEL': model
        }
    content = None
    replaced = False
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    for k in m:
        if m[k] and content.find(k) >= 0:
            replaced = True
            content = content.replace(k, m[k])
    if replaced:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
    return replaced

parser = argparse.ArgumentParser()
parser.add_argument('files', type=str, nargs='+',
                    help='files be replaced keywords')
parser.add_argument('--labels', '-l', type=str, help='to replace YOUR_LABELS')
parser.add_argument('--model', '-m', type=str, help='to replace YOUR_MODEL')
parser.add_argument('--modelin', '-i', type=str, help='to replace YOUR_MODELIN')
parser.add_argument('--modelout', '-o', type=str, help='to replace YOUR_MODELOUT')
parser.add_argument('--secret', '-s', type=str, help='to replace YOUR_CHANNEL_SECRET')
parser.add_argument('--token', '-t', type=str, help='to replace YOUR_CHANNEL_ACCESS_TOKEN')

args = parser.parse_args()

kws = dict(secret=args.secret,
           token=args.token,
           labels=args.labels,
           model=args.model,
           modelin=args.modelin,
           modelout=args.modelout)

for file in args.files:
    fs = glob(file)
    if fs:
        for f in fs:
            if os.path.isfile(f) and rep(f, **kws):
                print(f'{f} done')
