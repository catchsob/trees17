# basis:  pip instsall line-bot-sdk numpy flask pillow
# NAIVE:  pip install tensorflow
# REST:   pip install requests
# GRPC:   pip install grpcio tensorflow-serving-api
# CUSTOMVERTEX: pip install google-cloud-aiplatform

from io import BytesIO
import json
import os
import sys

from PIL import Image, ImageOps
import numpy as np
from flask import Flask, request, abort

from linebot.v3 import (
    WebhookHandler
)
from linebot.v3.exceptions import (
    InvalidSignatureError
)
from linebot.v3.messaging import (
    Configuration,
    ApiClient,
    MessagingApi,
    MessagingApiBlob,
    ReplyMessageRequest,
    TextMessage
)
from linebot.v3.webhooks import (
    MessageEvent,
    TextMessageContent,
    ImageMessageContent
)

app = Flask(__name__)

with open('env.json') as f:
    env = json.load(f)
configuration = Configuration(access_token=env['YOUR_CHANNEL_ACCESS_TOKEN'])
handler = WebhookHandler(env['YOUR_CHANNEL_SECRET'])
with ApiClient(configuration) as api_client:
    line_bot_api = MessagingApi(api_client)
    line_bot_blob_api = MessagingApiBlob(api_client)

res = env.get('YOUR_RESOLUTION', 448)
label_file = env.get('YOUR_LABELS')
mode = env.get('YOUR_MODE', 'NAIVE')  # NAIVE|REST|GRPC|CUSTOMVERTEX|ADV|FULL
if mode in ['NAIVE', 'REST', 'GRPC', 'CUSTOMVERTEX', 'ADV', 'FULL']:
    print(f'in {mode} mode')
else:
    sys.exit('unknown mode!')

if mode == 'REST':
    ssl = 's' if env.get('YOUR_REST_SSL', False) else ''
    rest_host = env.get('YOUR_REST_HOST', '127.0.0.1')
    rest_port = env.get('YOUR_REST_PORT', 8051)
    model_name = env.get('YOUR_MODEL_NAME', '')
    rest_url = f'http{ssl}://{rest_host}:{rest_port}/v1/models/{model_name}:predict'
elif mode == 'GRPC':
    import grpc
    grpc_host = env.get('YOUR_GRPC_HOST', '127.0.0.1')
    grpc_port = env.get('YOUR_GRPC_PORT', 8050)
    grpc_url = f'{grpc_host}:{grpc_port}'
    if env.get('YOUR_GRPC_SSL', False):
        channel = grpc.secure_channel(grpc_url, grpc.ssl_channel_credentials())
    else:
        channel = grpc.insecure_channel(grpc_url)
    model_name = env.get('YOUR_MODEL_NAME', 'default')
    model_in = env.get('YOUR_MODEL_IN', '')
    model_out = env.get('YOUR_MODEL_OUT', '')
elif mode =='CUSTOMVERTEX':
    from google.cloud import aiplatform
    res = 200
    gcp_service = env.get('YOUR_GCP_SERVICE')
    endpoint = env.get('YOUR_CUSTOMVERTEX_ENDPOINT')
    region = env.get('YOUR_CUSTOMVERTEX_REGION')
    project = env.get('YOUR_CUSTOMVERTEX_PROJ')
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = gcp_service
    aiplatform.init(project=project, location=region)
    endpoint = aiplatform.Endpoint(endpoint)
    if not project or not endpoint:
        sys.exit('environment error!')
elif mode == 'ADV':
    pass
elif mode == 'FULL':
    pass
else:  # default NAIVE
    from tensorflow.keras.models import load_model
    model = load_model(env['YOUR_MODEL_FILE'], compile=False)


@app.route("/callback", methods=['POST'])
def callback():
    # get X-Line-Signature header value
    signature = request.headers['X-Line-Signature']

    # get request body as text
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)

    # handle webhook body
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        app.logger.info("Invalid signature. Please check your channel access token/channel secret.")
        abort(400)

    return 'OK'


@handler.add(MessageEvent, message=TextMessageContent)
def handle_message(event):
    line_bot_api.reply_message_with_http_info(
        ReplyMessageRequest(
            reply_token=event.reply_token,
            messages=[TextMessage(text=event.message.text)]
        )
    )


@handler.add(MessageEvent, message=ImageMessageContent)
def handle_message(event):
    classify_fns = {'NAIVE': _classify,
                    'REST': _classify_rest,
                    'GRPC': _classify_grpc,
                    'CUSTOMVERTEX': _classify_customvertex,
                    'ADV': _classify_adv,
                    'FULL': _classify_full }

    message_content = line_bot_blob_api.get_message_content(message_id=event.message.id)
    image = _preprocess(Image.open(BytesIO(message_content)))
    label = classify_fns[mode](image)
    
    line_bot_api.reply_message(
        ReplyMessageRequest(
            reply_token=event.reply_token,
            messages=[TextMessage(text=label)]
        )
    )

def _preprocess(image):
    image = ImageOps.fit(image, (res, res))
    image = (np.expand_dims(image, axis=0)/255.).astype(np.float32)
    return image

def _classify(image):  # NAIVE mode
    prediction = model.predict(image)
    p = np.argmax(prediction)
    return _labels(p)

def _classify_rest(image):  # REST mode
    import requests

    headers = {"content-type": "application/json"}
    data = json.dumps({"instances": image.tolist()})
    r = requests.post(rest_url, headers=headers, data=data)
    p = np.argmax(r.json()['predictions'])
    return _labels(p)

def _classify_grpc(image):  # GRPC_MODE
    from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc
    from tensorflow import make_tensor_proto

    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    req = predict_pb2.PredictRequest()
    req.model_spec.name = model_name
    req.inputs[model_in].CopyFrom(make_tensor_proto(image))
    r = stub.Predict(req, 13.0)
    p = np.argmax(r.outputs[model_out].float_val)
    return _labels(p)

def _classify_adv(image):  # ADV_MODE
    from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc
    from tensorflow.core.framework import types_pb2
    from tensorflow.core.framework.tensor_shape_pb2 import TensorShapeProto
    from tensorflow.core.framework.tensor_pb2 import TensorProto

    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    req = predict_pb2.PredictRequest()
    req.model_spec.name = model_name
    d = [TensorShapeProto.Dim(size=x) for x in image.shape]
    t_p = TensorProto(dtype=types_pb2.DT_FLOAT, tensor_shape=TensorShapeProto(dim=d),
                      tensor_content=image.tobytes())
    req.inputs[model_in].CopyFrom(t_p)
    r = stub.Predict(req, 13.0)
    p = np.argmax(r.outputs[model_out].float_val)
    return _labels(p)

def _classify_customvertex(image):
    prediction = endpoint.predict(instances=image.tolist())  # instances < 1.5M
    p = np.ndarray.item(np.argmax(prediction.predictions, axis=1))
    return _labels(p)

def _classify_full(image):  # FULL_MODE
    pass

def _labels(prediction):
    with open(label_file, 'r', encoding='utf-8') as f:
        labels = f.read().split()
    return labels[prediction] if 0 <= prediction < len(labels) else '什麼樹？'


if __name__ == "__main__":
    app.run()
