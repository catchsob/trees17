import os
import argparse

from PIL import Image
import numpy as np

def preprocess_image(f, res=448, expand=False, precision='float32'):
    img = Image.open(f)
    img = img.resize((res, res))
    data = np.array(img.getdata())
    img = data.reshape(1, *img.size, 3) if expand else data.reshape(*img.size, 3)
    img = img.astype(precision)
    img /= 255.
    
    return img
    
parser = argparse.ArgumentParser()
parser.add_argument('-l', '--labels', type=str)
parser.add_argument('model', type=str)
parser.add_argument('tree', type=str)
args = parser.parse_args()

# inference for Keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.models import load_model
model = load_model(args.model)
img = preprocess_image(args.tree, expand=True)
pred = np.argmax(model.predict(img)[0])

if args.labels:
    try:
        with open(args.labels, encoding='utf-8') as f:
            pred = f.readlines()[pred].strip()
    except:
        pass

print(pred)