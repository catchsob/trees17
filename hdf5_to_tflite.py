import os
import argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

def h5_to_tfl(h5, tfl=None, res=200):
    savedmodel = os.path.splitext(h5)[0]
    model_h5 = tf.keras.models.load_model(h5)
    model_h5.save(savedmodel, save_format='tf')
    model_tf = tf.saved_model.load(savedmodel)
    
    concrete_func = model_tf.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    if None in concrete_func.inputs[0].shape.as_list()[1:-1]:
        concrete_func.inputs[0].set_shape([1, res, res, 3])
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    tflite_model = converter.convert()
    
    if not tfl:
        tfl = os.path.extsep.join([os.path.splitext(h5)[0], 'tflite'])    
    with open(tfl, 'wb') as f:
        f.write(tflite_model)
        
    return tfl

parser = argparse.ArgumentParser()
parser.add_argument('HDF5', type=str)
parser.add_argument('--tflite', type=str, default=None)
parser.add_argument('--resolution', type=int, default=200)

args = parser.parse_args()

a = h5_to_tfl(args.HDF5, args.tflite, args.resolution)
if a:
    print(f'{args.HDF5} converts to {a} done.')
else:
    print(f'{args.HDF5} converts failed!')
