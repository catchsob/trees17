import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('HDF5', type=str)
parser.add_argument('SAVEDMODEL', type=str, nargs='?')

args = parser.parse_args()
if not args.SAVEDMODEL:
    args.SAVEDMODEL = os.path.splitext(args.HDF5)[0]
if not os.path.isdir(args.SAVEDMODEL):
    os.mkdir(args.SAVEDMODEL)
    
from tensorflow.keras.models import load_model

load_model(args.HDF5).save(filepath=args.SAVEDMODEL, save_format='tf')
