import os
import sys
import argparse

import numpy as np
from PIL import Image


def crop(img):
    w, h = img.size
    if w == h:
        return img
    if h > w:
        d = int((h - w) / 2)
        return img.crop((0, d, w, h-d))
    d = int((w - h) / 2)
    return img.crop((d, 0, w-d, h))

def preprocess_image(images, res=448, expand=False, precision='float32', pn1=False):
    imgshape = (res, res)
    fs = images.split(',')
    imgs = np.empty((len(fs), *imgshape, 3), dtype=precision)
    for i, f in enumerate(fs):
        img = Image.open(f)
        img = crop(img)
        img = img.resize(imgshape)
        imgs[i] = np.reshape(img.getdata(), (*imgshape, 3))
    imgs = imgs / 127.5 - 1 if pn1 else imgs / 255.
    return imgs

def infer_h5(model, imgs):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    from tensorflow.keras.models import load_model
    
    model = load_model(model)
    preds = model.predict(imgs)
    return np.argmax(preds, axis=1)

def infer_tflite(model, imgs):
    import tflite_runtime.interpreter as tflite
    
    interpreter = tflite.Interpreter(model_path=model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    preds = []
    for img in imgs:
        interpreter.set_tensor(input_details[0]['index'], np.expand_dims(img, axis=0))
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])    
        results = np.squeeze(output_data)
        preds.append(results.argsort()[-5:][::-1][0])
    return preds

def infer_trt(model, imgs):
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    sys.path.append('/usr/src/tensorrt/samples/python')
    import common
    
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    with open(model, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    inputs, outputs, bindings, stream = common.allocate_buffers(engine)
    preds = []
    for img in imgs:
        inputs[0].host = np.reshape(img, [-1])
        with engine.create_execution_context() as context:
            trt_outputs = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
        pred = trt_outputs[0].reshape(1, -1)
        preds.append(np.argmax(pred[0]))
    return preds

def label(labels, preds):
    if labels:
        try:
            with open(labels, encoding='utf-8') as f:
                lbs = f.readlines()
            return [lbs[pred].strip() for pred in preds]
        except:
            pass
    return preds.tolist() if isinstance(preds, np.ndarray) else preds

infermap = {'h5': infer_h5, 'tflite': infer_tflite, 'trt': infer_trt}
parser = argparse.ArgumentParser()
parser.add_argument('model', type=str)
parser.add_argument('trees', type=str, help='image files of trees to be classified, seperated by common')
parser.add_argument('-f', '--model_format', type=str, choices=infermap.keys(), help='format of model')
parser.add_argument('-l', '--labels', type=str, help='file for prediction-label mapping')
parser.add_argument('-p', '--precision', type=str, default='float32', choices=['float16', 'float32'], help='default float32')
args = parser.parse_args()
    
imgs = preprocess_image(args.trees, precision=args.precision)

if args.model_format:
    m = args.model_format
else:
    ext = args.model.split(os.path.extsep)[-1]
    m = ext if ext in infermap else None
    
if m:
    preds = infermap[m](args.model, imgs)
    print(label(args.labels, preds))
