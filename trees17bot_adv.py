import io
import json

import grpc
import numpy as np
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc
from tensorflow.core.framework import types_pb2
from tensorflow.core.framework.tensor_shape_pb2 import TensorShapeProto
from tensorflow.core.framework.tensor_pb2 import TensorProto

from PIL import Image, ImageOps
from flask import Flask, request, abort

from linebot import (
    LineBotApi, WebhookHandler
)
from linebot.exceptions import (
    InvalidSignatureError
)
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage, ImageMessage
)

app = Flask(__name__)

line_bot_api = LineBotApi('YOUR_CHANNEL_ACCESS_TOKEN')
handler = WebhookHandler('YOUR_CHANNEL_SECRET')
label = 'YOUR_LABELS'
res = 448
grpcurl = 'YOUR_HOST:YOUR_PORT'
ssl = False
model = 'YOUR_MODELNAME'
modelin = 'YOUR_MODELIN'
modelout = 'YOUR_MODELOUT'
if ssl:
    channel = grpc.secure_channel(grpcurl, grpc.ssl_channel_credentials())
else:
    channel = grpc.insecure_channel(grpcurl)

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
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=event.message.text))


@handler.add(MessageEvent, message=ImageMessage)
def handle_message(event):
    message_id = event.message.id
    message_content = line_bot_api.get_message_content(message_id)
        
    b = b''
    for chunk in message_content.iter_content():
        b += chunk
    img = Image.open(io.BytesIO(b))
    r = classify_grpc(img)
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=r))


def classify_grpc(img):
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
    p = np.argmax(r.outputs[modelout].float_val)
    with open(label, encoding='utf-8') as f:
        labels = f.read().split()
    return labels[p] if 0 <= p < len(labels) else 'unknown'


if __name__ == "__main__":
    app.run()
