import os
import argparse

from tensorflow.keras.models import load_model

parser = argparse.ArgumentParser()
parser.add_argument('HDF5', type=str)
parser.add_argument('SAVEDMODEL', type=str)

args = parser.parse_args()
if not os.path.isdir(args.SAVEDMODEL):
    os.mkdir(args.SAVEDMODEL)
load_model(args.HDF5).save(filepath=args.SAVEDMODEL, save_format='tf')
