import sys
import argparse

import numpy as np
from PIL import Image
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

# inference for TersorRT
sys.path.append('/usr/src/tensorrt/samples/python')
import common

CAT = ['榕樹', '白千層', '楓香', '台灣欒樹', '小葉欖仁', '大葉欖仁', '茄冬',
       '黑板樹', '大王椰子', '鳳凰木', '阿勃勒', '水黃皮', '樟樹', '苦楝']

def preprocess_image(f, res=448, expand=False, precision='float32'):
    img = Image.open(f)
    img = img.resize((res, res))
    data = np.array(img.getdata())
    img = data.reshape(1, *img.size, 3) if expand else data.reshape(*img.size, 3)
    img = img.astype(precision)
    img /= 255.
            
    return img

parser = argparse.ArgumentParser()
parser.add_argument('engine', type=str)
parser.add_argument('tree', type=str)
parser.add_argument('-l', '--labels', type=str)
parser.add_argument('-p', '--precision', type=str, default='float32', choices=['float16', 'float32'])
args = parser.parse_args()
    
img = preprocess_image(args.tree, precision=args.precision)

# inference for TensorRT
runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
with open(args.engine, "rb") as f:
    engine = runtime.deserialize_cuda_engine(f.read())
inputs, outputs, bindings, stream = common.allocate_buffers(engine)
inputs[0].host = np.reshape(img, [-1])
with engine.create_execution_context() as context:
    trt_outputs = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
pred = trt_outputs[0].reshape(1, -1)
pred = np.argmax(pred[0])

if args.labels:
    try:
        with open(args.labels, encoding='utf-8') as f:
            pred = f.readlines()[pred].strip()
    except:
        pass

print(pred)