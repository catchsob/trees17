import argparse

import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image', default='榕樹.jpg', help='image')
    parser.add_argument('-l', '--labels', type=str, help='the file name for label-name mapping, illegal file would be ignored')
    parser.add_argument('-m', '--model', default='trees17V1.tflite', help='.tflite model')
    args = parser.parse_args()

    interpreter = tflite.Interpreter(model_path=args.model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
    img = Image.open(args.image).resize((width, height))
    input_data = np.expand_dims(img, axis=0)
    floating_model = input_details[0]['dtype'] == np.float32
    if floating_model:
        input_data = input_data.astype('float32') / 255.
    interpreter.set_tensor(input_details[0]['index'], input_data)
    
    interpreter.invoke()
    
    output_data = interpreter.get_tensor(output_details[0]['index'])    
    results = np.squeeze(output_data)
    ans = results.argsort()[-5:][::-1][0]
    
    if args.labels:
        try:
            with open(args.labels, encoding='utf-8') as f:
                ans = f.readlines()[ans].strip()
        except:
            pass
    
    print(ans)
