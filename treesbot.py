import io
import json

import numpy as np
from tensorflow.keras.models import load_model
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

with open('env.json') as f:
    env = json.load(f)
line_bot_api = LineBotApi(env['YOUR_CHANNEL_ACCESS_TOKEN'])
handler = WebhookHandler(env['YOUR_CHANNEL_SECRET'])
model = load_model(env['YOUR_MODEL_FILE'])
label = env['YOUR_LABELS']

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
    r = classify(img)
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=r))


def classify(img):
    img = ImageOps.fit(img, model.input.shape[1:3])
    prediction = model.predict(np.expand_dims(img, axis=0)/255.)
    p = np.argmax(prediction)
    with open(label, encoding='utf-8') as f:
        labels = f.read().split()
    return labels[p] if 0 <= p < len(labels) else 'unknown'


if __name__ == "__main__":
    app.run()
