import os
import sys
import argparse
import json

import numpy as np
from PIL import Image, ImageOps


def preprocess_images(images, res=448, precision='float32', pn1=False):
    imgshape = (res, res)
    imgs = np.empty((len(images), *imgshape, 3), dtype=precision)
    for i, image in enumerate(images):
        img = Image.open(image)
        img = ImageOps.fit(img, imgshape)
        imgs[i] = np.reshape(img.getdata(), (*imgshape, 3))
    imgs = imgs / 127.5 - 1 if pn1 else imgs / 255.
    return imgs

def label(labels, preds):
    if labels:
        try:
            with open(labels, encoding='utf-8') as f:
                lbs = f.readlines()
            return [lbs[pred].strip() for pred in preds]
        except Exception as e:
            pass
    return preds.tolist() if isinstance(preds, np.ndarray) else preds

def dump_images(images, res=200, pn1=False):
    data = preprocess_images(images=images, res=res, pn1=pn1)    
    print(json.dumps({'instances': data.tolist()}))

def infer_vertex(images, res, pn1, proj, region, endp, serv):
    from google.cloud import aiplatform
    
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = serv
    aiplatform.init(project=proj, location=region)
    endpoint = aiplatform.Endpoint(endp)
    data = preprocess_images(images=images, res=res, pn1=pn1)
    prediction = endpoint.predict(instances=data.tolist())  # instances < 1.5M
    return np.argmax(prediction.predictions, axis=1)

def infer_automl(images, res, pn1, proj, region, endp, serv):
    import predict_image_classification_sample as p
    
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = serv
    for image in images:
        r = p.predict_image_classification_sample(project=proj,
                                                  endpoint_id=endp,
                                                  filename=image,
                                                  location=region,
                                                  api_endpoint=f'{region}-aiplatform.googleapis.com')


vertexmap = {'Vertex': infer_vertex, 'AutoML': infer_automl}
aiservingmap = list(vertexmap.keys())
parser = argparse.ArgumentParser()
parser.add_argument('trees', type=str, nargs='+',
                    help='image files of trees to be classified, seperated by common')
parser.add_argument('-a', '--aiserving', type=str, choices=aiservingmap, help='types of AI serving')
parser.add_argument('-d', '--dump', action='store_true', help='dump images in JSON for RESTful AI serving')
parser.add_argument('-l', '--labels', type=str, help='file for prediction-label mapping')
parser.add_argument('-r', '--resolution', type=int, default=448, help='resolution of images, default 448')
parser.add_argument('-p', '--project', type=str, help='your project ID of Vertex serving')
parser.add_argument('-s', '--service', type=str, help='your service file in JSON')
parser.add_argument('-e', '--endpoint', type=str, help='your endpoint ID of Vertex serving')
parser.add_argument('-R', '--region', type=str, help='your region of Vertex serving')


args = parser.parse_args()
if args.dump:
    dump_images(args.trees)

if args.aiserving in vertexmap:
    if not args.project:
        sys.exit('exit due to lack of your project ID')
    if not args.service:
        sys.exit('exit due to lack of your service file in JSON')
    if not args.endpoint:
        sys.exit('exit due to lack of your endpoint ID')
    if not args.region:
        sys.exit('exit due to lack of your region')
    
    preds = vertexmap[args.aiserving](images=args.trees,
                                      res=args.resolution,
                                      pn1=False,
                                      proj=args.project,
                                      region=args.region,
                                      endp=args.endpoint,
                                      serv=args.service)
    if args.aiserving == 'Vertex':
        print(label(args.labels, preds))
