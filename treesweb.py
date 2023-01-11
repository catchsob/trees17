import json
from io import BytesIO

from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image, ImageOps
import numpy as np


app = Flask(__name__)

with open('env.json') as f:
    env = json.load(f)
label = env['YOUR_LABELS']
model = env['YOUR_MODEL_NAME']
model_file = env['YOUR_MODEL_FILE']
model_local = None
res = 448
restssl = None  # rest
resturl = None  # rest
grpcssl = None  # grpc
grpcurl = None  # grpc
modelin = None  # grpc
modelout = None  # grpc
channel = None  # grpc

STYLE = '''
<style type="text/css">
    div, input {
        font-size: 4vw;
    }
    label {
        color: brown;
    }
    img {
        width: 100%;
    }
</style>
'''

@app.route("/treesl", methods=['GET'])
def local():
    return '<title>Trees Local</title><br>' + gen_page(api='local')


@app.route("/trees", methods=['GET', 'POST'])
def ask():
    pred, conf, filename = '', '', ''
    
    if request.method == 'POST':
        global restssl, resturl
    
        if None in [restssl, resturl]:
            restssl = 's' if env['YOUR_REST_SSL'] else ''
            resturl = f'http{restssl}://{env["YOUR_REST_HOST"]}:{env["YOUR_REST_PORT"]}/v1/models/{model}:predict'
        
        r = predict(api='rest')
        pred = r['prediction']
        conf = f"{r['confidence']:.4f}"
    
    page = f'''
    <title>Trees Web</title>
    <div id="select">
        <form method="post" enctype="multipart/form-data">
        Image: <input type="file" id="select_img" name="imagefile" accept="image/*, capture=camera">
        <input type="submit" value="確定">
        Prediction: <label id="pred">{pred}</label>/ Confidence: <label id="conf">{conf}</label>
        </form>
    </div>
    <br>'''

    return page + STYLE


@app.route("/treesr", methods=['GET'])
def ask_rest():
    return '<title>Trees REST</title><br>' + gen_page(api='rest')


@app.route("/treesg", methods=['GET'])
def ask_grpc():
    return '<title>Trees gRPC</title><br>' + gen_page(api='grpc')

        
def gen_page(api='rest'):
    apimap = {'rest': 'treesr', 'grpc': 'treesg', 'local': 'treesl'}
    
    # the ugly code below is for easy understanding, should be replaced by render_template()
    page1 = '''
    <div id="select">
        Image: <input type="file" id="select_img" name="imagefile" accept="image/*, capture=camera" onchange="preview()">
    </div>
    <div>
        Prediction: <label id="pred"></label>/ Confidence: <label id="conf"></label>
    </div>
    <br>
    <img id="preview_img" /><br>
    
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
                        console.log("Error:", error);
                    })
            };

            let file = document.getElementById("select_img").files[0];
            reader.readAsDataURL(file);
        }
    </script>
    '''

    return page1 + page2 + page3 + STYLE


@app.route("/treesl", methods=['POST'])
def predict_local():
    from tensorflow.keras.models import load_model

    global model_local
    
    data = {'prediction': None, 'confidence': None}
    file = request.files.get('imagefile')
    
    if not model_local:
        model_local = load_model(model_file)
    
    if file:
        filename = secure_filename(file.filename)
        print(f'got {filename}')
        img = file.read()
        img = Image.open(BytesIO(img))
        img = ImageOps.fit(img, (res, res))
        prediction = model_local.predict(np.expand_dims(img, axis=0)/255.)[0]
        p = np.argmax(prediction)
        with open(label, encoding='utf-8') as f:
            labels = f.read().split()
        data['prediction'] = labels[p] if 0 <= p < len(labels) else str(p)
        data['confidence'] = prediction[p].astype(float)
        
    return jsonify(data)


@app.route("/treesr", methods=['POST'])
def predict_rest():
    global restssl, resturl
    
    if None in [restssl, resturl]:
        restssl = 's' if env['YOUR_REST_SSL'] else ''
        resturl = f'http{restssl}://{env["YOUR_REST_HOST"]}:{env["YOUR_REST_PORT"]}/v1/models/{model}:predict'
    
    return jsonify(predict(api='rest'))


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

    return jsonify(predict(api='grpc'))


def predict(api='rest'):
    data = {'prediction': None, 'confidence': None}
    
    file = request.files.get('imagefile')
    
    if file:
        apimap = {'rest': classify_rest, 'grpc': classify_grpc}
        filename = secure_filename(file.filename)
        print(f'got {filename}')
        img = file.read()
        img = Image.open(BytesIO(img))
        img = ImageOps.fit(img, (res, res))
        data['prediction'], data['confidence'] = apimap[api](img)

    return data


def classify_grpc(img):
    import grpc
    from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc
    from tensorflow.core.framework import types_pb2
    from tensorflow.core.framework.tensor_shape_pb2 import TensorShapeProto
    from tensorflow.core.framework.tensor_pb2 import TensorProto
    
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
    return labels[p] if 0 <= p < len(labels) else str(p), r[p]


def classify_rest(img):
    import requests

    img = np.expand_dims(img, axis=0)/255.
    
    headers = {"content-type": "application/json"}
    data = json.dumps({"instances": img.tolist()})
    r = requests.post(resturl, headers=headers, data=data)
    
    r = np.squeeze(r.json()['predictions'])
    p = np.argmax(r)
    with open(label, encoding='utf-8') as f:
        labels = f.read().split()
    return labels[p] if 0 <= p < len(labels) else str(p), r[p]


if __name__ == '__main__':
    app.run()
