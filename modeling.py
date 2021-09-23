# -*- coding: utf-8 -*-
import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from gensim.models import FastText
from gensim.models.keyedvectors import KeyedVectors
import argparse
import io

def get_inputtext():
    dlg_list = []
    with io.open("data/textandembedding/보관/allmorptok_inputtextfile.txt", encoding='euc-kr', mode='r') as f:
        for a in f:
            dlg_list.append(a.strip().split('\t'))
    dlg_list = pd.DataFrame(dlg_list, columns=["dlg", "check"])
    return dlg_list

def visual_dlg(data):
    data['sentence'] = data["dlg"].apply(lambda x: len(x.split()))
    mean_seq_len = np.round(data['sentence'].mean()).astype(int)
    sns.distplot(tuple(data['sentence']), hist=True, kde=True, label='Sentence length')
    plt.axvline(x=mean_seq_len, color='k', linestyle='--', label=f'Sequence length mean:{mean_seq_len}')
    plt.title('Sentence length')
    plt.legend()
    plt.show()
    print(f"가장 긴 문장 내 단어의 수 : {data['sentence'].max()}")
    print(f"가장 짧은 문장 내 단어의 수 : {data['sentence'].min()}")
    print(f"평균 문장 내 단어의 수 : {mean_seq_len}")
def tokenizing_dlg(X):
    tok = Tokenizer()
    tok.fit_on_texts(X)
    vocab_size = len(t.word_index) + 1
    return tok,vocab_size

def seq_pad_dlg(t,X):
    dlg_sequence = t.texts_to_sequences(X)
    max_len = max(len(l) for l in dlg_sequence)
    dlg_padding = pad_sequences(dlg_sequence, padding='post', maxlen=max_len)
    return dlg_padding,max_len

def emb_matrix(t,ft):
    embedding_dim = 100
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, i in t.word_index.items():  # 훈련 데이터의 단어 집합에서 단어와 정수 인덱스를 1개씩 꺼내온다.
        temp = ft.wv[word]  # 단어(key) 해당되는 임베딩 벡터의 값(value)를 temp에 저장
        if temp is not None:  # 만약 temp가 None이 아니라면 임베딩 벡터의 값을 리턴받은 것이므로
            embedding_matrix[i] = temp  # embedding_matrix의 해당 단어 위치의 행에 벡터의 값을 저장한다.
    return embedding_matrix

def bilstm_modeling(vocab_size,embedding_dim,embedding_matrix,max_len,dlg_padding,y):
    model_bilstm = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], input_length=max_len),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=128, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=64, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=64)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model_bilstm.compile(loss='binary_crossentropy',
                         optimizer='adam',
                         metrics=['accuracy'])

    model_bilstm.summary()
    history = model_bilstm.fit(dlg_padding, y,
                               epochs=10,
                               batch_size=128)
    return model_bilstm,history
if __name__ == '__main__':
    embedding_dim = 100

    dlg_list = get_inputtext()
    visual_dlg(dlg_list)
    X = dlg_list["dlg"]
    y = pd.to_numeric(dlg_list["check"])
    t,vocab_size = tokenizing_dlg(X)
    dlg_padding,max_len=seq_pad_dlg(t,X)
    ft = FastText.load('data/textandembedding/allmorptok_emmbedingtext_130_fasttext')
    embedding_matrix = emb_matrix(t,ft)
    model_bilstm, history = bilstm_modeling(vocab_size,embedding_dim,embedding_matrix,max_len,dlg_padding,y)

    ft = FastText.load('data/textandembedding/allmorptok_emmbedingtext_130_fasttext')
