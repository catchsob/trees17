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
    MessageEvent, TextMessage, TextSendMessage, ImageMessage, QuickReply, QuickReplyButton, URIAction,
)

app = Flask(__name__)

with open('env.json') as f:
    env = json.load(f)
line_bot_api = LineBotApi(env['YOUR_CHANNEL_ACCESS_TOKEN'])
handler = WebhookHandler(env['YOUR_CHANNEL_SECRET'])
label = env['YOUR_LABELS']
res = 448
grpcurl = f'{env["YOUR_GRPC_HOST"]}:{env["YOUR_GRPC_PORT"]}'
ssl = env['YOUR_GRPC_SSL']
model = env['YOUR_MODEL_NAME']
modelin = env['YOUR_MODEL_IN']
modelout = env['YOUR_MODEL_OUT']
web = env['YOUR_WEB_URI']
tfjs = env['YOUR_TFJS_URI']
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
    msg = TextSendMessage(text=event.message.text)
    if event.message.text == 'web':
        items = [QuickReplyButton(action=URIAction(label='Web', uri=web)),
                 QuickReplyButton(action=URIAction(label='TensorFlow.js', uri=tfjs))]
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
