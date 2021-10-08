# -*- coding: utf-8 -*
from konlpy.tag import Mecab

from flask import Flask,request
import requests
from transformers import BertTokenizerFast
from transformers import BertForSequenceClassification
import torch
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
tokenizer = Tokenizer()
import pandas as pd
import pickle
from konlpy.tag import Mecab
import json


mecab= Mecab("C:\\mecab\\mecab-ko-dic")


app = Flask(__name__)
api_url='https://api.telegram.org'
token="2035664374:AAEgpPe6OTaYziyECyc8SU9W8mWQ9GBGuX0"
chat_id= "2083695627"
dlg_list=[]
cloth_type_dict = {"SW":"스웨터","SK":"스커트","CT":"코트","SE":"신발","PT":"팬츠","JP":"후드","CD":"가디건","KN":"저지","OP":"원피스","BL":"블라우스","SH":"셔츠","JK":"자켓","VT":"조끼"}
# botoom_list = ["스커트","팬츠"]
# top_list=["스웨터","셔츠","블라우스"]
# outer_list=["코트","가디건","조끼","자켓","후드","저지"]
id_dict={0:"sw_model_lstm.h5",1:"sk_model_lstm.h5",2:"ct_model_lstm.h5",3:"se_model_rnn.h5",4:"pt_model_lstm.h5",5:"jp_model_rnn.h5",6:"cd_model_lstm.h5",7:"kn_model_lstm.h5",8:"op_model_gru.h5",9:"bl_model_rnn.h5",10:"sh_model_rnn.h5",11:"jk_model_lstm.h5",12:"vt_model_rnn.h5"}
type_dict={0:"SW",1:"SK",2:"CT",3:"SE",4:"PT",5:"JP",6:"CD",7:"KN",8:"OP",9:"BL",10:"SH",11:"JK",12:"VT"}

bert_model_name = 'beomi/kcbert-large'
type_classify_model = BertForSequenceClassification.from_pretrained("C:/Users/tlsgh/PycharmProjects/pythonProject1/model/checkpoint-1373/")
tokenizer = BertTokenizerFast.from_pretrained(bert_model_name)


def recommend_cloth(user_text):
    encoded_contents = tokenizer.encode_plus(
        user_text,
        add_special_tokens=True,
        max_length= tokenizer.model_max_length,
        return_token_type_ids=False,
        return_attention_mask=True,
        return_tensors="pt",
        padding="max_length",
        truncation=True
    )

    inputs = encoded_contents['input_ids']
    masks = encoded_contents['attention_mask']

    # inputs = torch.tensor(input_ids)
    # masks = torch.tensor(attention_masks)

    # b_input_ids = inputs.to(device)
    # b_input_mask = masks.to(device)
    with torch.no_grad():
        outputs = type_classify_model(inputs, attention_mask=masks)
    logits = outputs[0]
    logits = logits.detach().cpu().numpy()
    cloth_type_number=np.argmax(logits[0]) - 1
    recommend_model = tf.keras.models.load_model("model/typeclassify/"+id_dict[cloth_type_number])
    with open('model/tokenizer/{}_tokenizer'.format(type_dict[cloth_type_number]), 'rb') as handle:
        keras_tokenizer = pickle.load(handle)
    morph_text = mecab.morphs(user_text)
    morph_text=[morph_text]
    token_text = keras_tokenizer.texts_to_sequences(morph_text)
    pad_text = pad_sequences(token_text, maxlen=300)
    logits2=recommend_model.predict([pad_text])
    cloth_number = np.argmax(logits2) - 1
    cloth_name=""
    if len(str(cloth_number)) == 1:
        cloth_name=type_dict[cloth_type_number]+"-"+"00"+ str(cloth_number)
    elif len(str(cloth_number)) == 2:
        cloth_name=type_dict[cloth_type_number]+"-"+"0"+ str(+cloth_number)
    elif len(str(cloth_number)) == 3:
        cloth_name = type_dict[cloth_type_number] + "-" + str(cloth_number)

    return cloth_name
@app.route('/')
def home():
    return 'Hello, World!'
@app.route('/write')
def write():
    return 'Hello, World!'
@app.route('/send')
def send():
    text = requests.args.get('message')
    requests.get(f'{api_url}/bot{token}/sendMessage?chat_id={chat_id}&text={text}')
    return 'Hello, World!'


@app.route(f'/chatbot',methods=['POST'])
def telagram():
    if "edited" in list(request.get_json().keys())[1]:
        chat_id = request.get_json().get("edited_message").get('from').get('id')
        user_text = request.get_json()["edited_message"]['text']
    else:
        chat_id = request.get_json().get("message").get('from').get('id')
        user_text = request.get_json()["message"]['text']
    dlg_list.append(user_text)
    print(user_text)

    if "start" in user_text:
        com_text = "어서오세요. 코디봇입니다. 어떤 옷을 추천해드릴까요?\nex) 주말에 친척언니 결혼식 가는데 격식 있고 화사한 원피스 추천해주세요"
        requests.get(f'{api_url}/bot{token}/sendMessage?chat_id={chat_id}&text={com_text}')
    elif "추천" in user_text:
        url = "https://api.telegram.org/bot{}/sendPhoto".format(token)
        rec_cloth=recommend_cloth(user_text).strip()
        img = "data/image/{}.jpg".format(rec_cloth)
        files = {'photo': open(img, 'rb')}
        data = {'chat_id': chat_id}
        r = requests.post(url, files=files, data=data)
        com_text = "{} 추천드렸습니다.\n다른 옷을 추천해드릴까요?".format(cloth_type_dict[rec_cloth.split('-')[0]])
        requests.get(f'{api_url}/bot{token}/sendMessage?chat_id={chat_id}&text={com_text}')
        recing = 0
    elif "네" in user_text or "넹" in user_text or "엉" in user_text or "좋아" in user_text:
        com_text = "추천 받고 싶은 다른 옷 하나와 오늘 기분 및 상황을 적어주세요"
        requests.get(f'{api_url}/bot{token}/sendMessage?chat_id={chat_id}&text={com_text}')
        recommending = 1
    elif "아니" in user_text or "싫어" in user_text or "괜찮아" in user_text:
        com_text = "넵. 감사합니다. 다음번에 뵈요.."
        requests.get(f'{api_url}/bot{token}/sendMessage?chat_id={chat_id}&text={com_text}')
    else:
        com_text = "추천 받고 싶은 옷 하나와 오늘 기분 및 상황을 다시 채팅보내주세요."
        requests.get(f'{api_url}/bot{token}/sendMessage?chat_id={chat_id}&text={com_text}')
    return '',200


if __name__ == '__main__':
    app.run(debug=True)