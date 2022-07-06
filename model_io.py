from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('PATH', type=str, help='SavedModel path or HDF5 file name')
args = parser.parse_args()

from tensorflow.keras.models import load_model

model = load_model(args.PATH)

print(f' input name: {model.layers[0].name}')
print(f'output name: {model.layers[-1].name}')
