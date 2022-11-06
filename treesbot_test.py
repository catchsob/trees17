import json
from io import BytesIO

from flask import Flask, request, abort
from PIL import Image, ImageOps
import numpy as np
from linebot import (
    LineBotApi, WebhookHandler
)
from linebot.exceptions import (
    InvalidSignatureError
)
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage, ImageMessage, QuickReply, QuickReplyButton,
    URIAction, CameraAction, CameraRollAction
)


app = Flask(__name__)

with open('env.json') as f:
    env = json.load(f)
line_bot_api = LineBotApi(env['YOUR_CHANNEL_ACCESS_TOKEN'])
handler = WebhookHandler(env['YOUR_CHANNEL_SECRET'])
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
web = env['YOUR_WEB_URI']
tfjs = env['YOUR_TFJS_URI']


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
        print("Invalid signature. Please check your channel access token/channel secret.")
        abort(400)

    return 'OK'


@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    msg = TextSendMessage(text=event.message.text)
    if event.message.text == 'fn':
        items = [QuickReplyButton(action=CameraAction(label='Cam')),
                 QuickReplyButton(action=CameraRollAction(label='Pic')),
                 QuickReplyButton(action=URIAction(label='Web', uri=env['YOUR_WEB_URI'])),
                 QuickReplyButton(action=URIAction(label='LIFF', uri='https://liff.line.me/'+env['YOUR_LIFF_ID'])),
                 QuickReplyButton(action=URIAction(label='TF.js', uri=env['YOUR_TFJS_URI']))
                ]
        msg.quickReply = QuickReply(items=items)
    line_bot_api.reply_message(
        event.reply_token,
        msg)


@handler.add(MessageEvent, message=ImageMessage)
def handle_message(event):
    message_id = event.message.id
    message_content = line_bot_api.get_message_content(message_id)
        
    b = b''
    for chunk in message_content.iter_content():
        b += chunk
    img = Image.open(BytesIO(b))
    img = ImageOps.fit(img, (res, res))
    r = classify_rest(img)
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=r))


def classify_rest(img):
    import requests
    
    global restssl, resturl
    
    if None in [restssl, resturl]:
        restssl = 's' if env['YOUR_REST_SSL'] else ''
        resturl = f'http{restssl}://{env["YOUR_REST_HOST"]}:{env["YOUR_REST_PORT"]}/v1/models/{model}:predict'
    
    img = np.expand_dims(img, axis=0)/255.
    
    headers = {"content-type": "application/json"}
    data = json.dumps({"instances": img.tolist()})
    r = requests.post(resturl, headers=headers, data=data)
    
    r = np.squeeze(r.json()['predictions'])
    p = np.argmax(r)
    with open(label, encoding='utf-8') as f:
        labels = f.read().split()
    return labels[p] if 0 <= p < len(labels) else 'unknown'


def classify_grpc(img):
    import grpc
    from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc
    from tensorflow.core.framework import types_pb2
    from tensorflow.core.framework.tensor_shape_pb2 import TensorShapeProto
    from tensorflow.core.framework.tensor_pb2 import TensorProto
    
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
    
    img = (np.expand_dims(img, axis=0)/255.).astype(np.float32)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    req = predict_pb2.PredictRequest()
    req.model_spec.name = model
    d = [TensorShapeProto.Dim(size=x) for x in img.shape]
    t_p = TensorProto(dtype=types_pb2.DT_FLOAT, tensor_shape=TensorShapeProto(dim=d),
                      tensor_content=img.tobytes())
    req.inputs[modelin].CopyFrom(t_p)
    r = stub.Predict(req, 13.0)
    p = np.argmax(r.outputs[modelout].float_val)
    with open(label, encoding='utf-8') as f:
        labels = f.read().split()
    return labels[p] if 0 <= p < len(labels) else 'unknown'


if __name__ == "__main__":
    app.run()
