import io
import json

from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from PIL import Image, ImageOps
import numpy as np


app = Flask(__name__)

with open('env.json') as f:
    env = json.load(f)
label = env['YOUR_LABELS']
model = env['YOUR_MODEL_NAME']
res = 448
restssl = None  # rest
resturl = None  # rest
grpcssl = None  # grpc
grpcurl = None  # grpc
modelin = None  # grpc
modelout = None  # grpc
channel = None  # grpc

@app.route("/trees", methods=['GET'])
def ask():
    page = '''
    <div id="select">Image:
    <input type="file" id="select_img" name="imagefile" accept="image/*, capture=camera" onchange="preview()">
    </div>
    <div>Prediction: <label id="pred"></label>/ Confidence: <label id="conf"></label></div>
    <br>
    <img id="preview_img" /><br>

    <style type="text/css">
        div, input {
            font-size: 4vw;
        }
        label {
            color: brown;
        }
        img {
            width: 80%;
        }
    </style>
    
    <script language="javascript">    
        function preview() {
            if (!window.FileReader) {
                console.log("no preview functionality supported by your browser!");
                return;
            }
            
            let reader = new FileReader();
            reader.onload = function (event) {
                let img = document.getElementById("preview_img");
                img.src = event.target.result;
                const payload = new FormData();
                payload.append('imagefile', file, file.name);
                fetch("trees", {method: "POST", body: payload})
                    .then((response) => response.json())
                    .then((data) => {
                        document.getElementById("pred").innerHTML = data["prediction"]
                        document.getElementById("conf").innerHTML = data["confidence"].toPrecision(4)
                        console.log(data); 
                    })
                    .catch((error) => {
                        console.log("Error: ${error}");
                    })
            };

            let file = document.getElementById("select_img").files[0];
            reader.readAsDataURL(file);
        }
    </script>
    '''
    
    return page

@app.route("/treesr", methods=['GET'])
def ask_rest():
    return gen_page(api='rest')
    
@app.route("/treesg", methods=['GET'])
def ask_grpc():
    return gen_page(api='grpc')

def gen_page(api='rest'):
    apimap = {'rest': 'treesr', 'grpc': 'treesg'}
    
    page1 = '''
    <div id="select">Image:
    <input type="file" id="select_img" name="imagefile" accept="image/*, capture=camera" onchange="preview()">
    </div>
    <div>Prediction: <label id="pred"></label>/ Confidence: <label id="conf"></label></div>
    <br>
    <img id="preview_img" /><br>

    <style type="text/css">
        div, input {
            font-size: 4vw;
        }
        label {
            color: brown;
        }
        img {
            width: 80%;
        }
    </style>
    
    <script language="javascript">    
        function preview() {
            if (!window.FileReader) {
                console.log("no preview functionality supported by your browser!");
                return;
            }
            
            let reader = new FileReader();
            reader.onload = function (event) {
                let img = document.getElementById("preview_img");
                img.src = event.target.result;
                const payload = new FormData();
                payload.append("imagefile", file, file.name);
                fetch('''
    page2 = f'"{apimap[api]}"'
    page3 = ''', {method: "POST", body: payload})
                    .then((response) => response.json())
                    .then((data) => {
                        document.getElementById("pred").innerHTML = data["prediction"]
                        document.getElementById("conf").innerHTML = data["confidence"].toPrecision(4)
                        console.log(data); 
                    })
                    .catch((error) => {
                        console.log("Error: ${error}");
                    })
            };

            let file = document.getElementById("select_img").files[0];
            reader.readAsDataURL(file);
        }
    </script>
    '''
    
    return page1 + page2 + page3

@app.route("/treesr", methods=['POST'])
def predict_rest():
    global restssl, resturl
    
    if None in [restssl, resturl]:
        restssl = 's' if env['YOUR_REST_SSL'] else ''
        resturl = f'http{restssl}://{env["YOUR_REST_HOST"]}:{env["YOUR_REST_PORT"]}/v1/models/{env["YOUR_MODEL_NAME"]}:predict'
    
    return predict(api='rest')
    
@app.route("/treesg", methods=['POST'])
def predict_grpc():
    import grpc
    
    global grpcssl, grpcurl, channel, modelin, modelout
    
    if None in [grpcssl, grpcurl, channel, modelin, modelout]:
        grpcurl = f'{env["YOUR_GRPC_HOST"]}:{env["YOUR_GRPC_PORT"]}'
        grpcssl = env['YOUR_GRPC_SSL']
        modelin = env['YOUR_MODEL_IN']
        modelout = env['YOUR_MODEL_OUT']
        if grpcssl:
            channel = grpc.secure_channel(grpcurl, grpc.ssl_channel_credentials())
        else:
            channel = grpc.insecure_channel(grpcurl)

    return predict(api='grpc')
    
def predict(api='rest'):
    data = {'prediction': None, 'confidence': None}
    
    if request.method == 'POST':
        file = request.files.get('imagefile')
        if file:
            apimap = {'rest': classify_rest, 'grpc': classify_grpc}
            filename = secure_filename(file.filename)
            print(f'got {filename}')
            img = file.read()
            img = Image.open(io.BytesIO(img))
            data['prediction'], data['confidence'] = apimap[api](img)

    return jsonify(data)

def classify_grpc(img):
    import grpc
    from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc
    from tensorflow.core.framework import types_pb2
    from tensorflow.core.framework.tensor_shape_pb2 import TensorShapeProto
    from tensorflow.core.framework.tensor_pb2 import TensorProto
    
    img = ImageOps.fit(img, (res, res))
    img = (np.expand_dims(img, axis=0)/255.).astype(np.float32)
    
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    req = predict_pb2.PredictRequest()
    req.model_spec.name = model
    d = [TensorShapeProto.Dim(size=x) for x in img.shape]
    t_p = TensorProto(dtype=types_pb2.DT_FLOAT, tensor_shape=TensorShapeProto(dim=d),
                      tensor_content=img.tobytes())
    req.inputs[modelin].CopyFrom(t_p)
    r = stub.Predict(req, 13.0)
    
    r = np.array(r.outputs[modelout].float_val)
    p = np.argmax(r)
    with open(label, encoding='utf-8') as f:
        labels = f.read().split()
    return labels[p] if 0 <= p < len(labels) else 'unknown', r[p]

def classify_rest(img):
    import requests

    img = ImageOps.fit(img, (res, res))
    img = np.expand_dims(img, axis=0)/255.
    
    headers = {"content-type": "application/json"}
    data = json.dumps({"instances": img.tolist()})
    r = requests.post(resturl, headers=headers, data=data)
    
    r = np.squeeze(r.json()['predictions'])
    p = np.argmax(r)
    with open(label, encoding='utf-8') as f:
        labels = f.read().split()
    return labels[p] if 0 <= p < len(labels) else 'unknown', r[p]

if __name__ == '__main__':
    app.run()
