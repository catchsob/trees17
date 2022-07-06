import argparse
import json

import numpy as np
import requests
import grpc
from PIL import Image
from tensorflow import make_tensor_proto
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc

def preprocess(filename, res=400):
    img = Image.open(filename)
    img = img.resize((res, res))
    data = np.array(img.getdata())
    img = data.reshape(1, *img.size, 3)
    img = img.astype('float32')
    img /= 255.
    return img

def test_rest(img, model, host, port=8501, ssl=False):
    headers = {"content-type": "application/json"}
    data = json.dumps({"instances": img.tolist()})
    r = requests.post(f'http{"s" if ssl else ""}://{host}:{port}/v1/models/{model}:predict',
                      headers=headers, data=data)
    return np.argmax(r.json()['predictions'])
    
def test_grpc(img, model, host, modelin='input_1', modelout='dense_1', port=8500, ssl=False):
    if ssl:
        channel = grpc.secure_channel(f'{host}:{port}', grpc.ssl_channel_credentials())
    else:
        channel = grpc.insecure_channel(f'{host}:{port}')
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = model
    request.inputs[modelin].CopyFrom(make_tensor_proto(img))
    r = stub.Predict(request, 13.0)
    return np.argmax(r.outputs[modelout].float_val)

parser = argparse.ArgumentParser()
parser.add_argument("--protocol", type=str, default='gRPC', choices=['gRPC', 'REST'])
parser.add_argument("--ssl", action='store_true', default=False)
parser.add_argument("--host", type=str, default='localhost', help='default localhost')
parser.add_argument("--port", type=int, help='default 8500 for gRPC, 8501 for REST API')
parser.add_argument("--labels", type=str, help='the file name for label-name mapping, illegal file would be ignored')
parser.add_argument("--input", type=str, default='input_1', help='input layer name of model')
parser.add_argument("--output", type=str, default='dense_1', help='output layer name of model')
parser.add_argument("MODEL", type=str, help='model name, must be same with in server')
parser.add_argument("PIC", type=str, help='the file name of image to be classified')

args = parser.parse_args()

img = preprocess(args.PIC)
if args.protocol == 'REST':
    r = test_rest(img=img, model=args.MODEL, host=args.host,
                  port=args.port or 8501, ssl=args.ssl)
elif args.protocol == 'gRPC':
    r = test_grpc(img=img, model=args.MODEL, host=args.host,
                  modelin=args.input, modelout=args.output,
                  port=args.port or 8500, ssl=args.ssl)

if args.labels:
    try:
        with open(args.labels, encoding='utf-8') as f:
            r = f.readlines()[r].strip()
    except:
        pass
print(r)
