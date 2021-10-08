# -*- coding: utf-8 -*-
import pandas as pd
from collections import Counter
from tensorflow.keras.preprocessing.text import Tokenizer
import pickle
from eunjeon import Mecab

type_dict={0:"SW",1:"SK",2:"CT",3:"SE",4:"PT",5:"JP",6:"CD",7:"KN",8:"OP",9:"BL",10:"SH",11:"JK",12:"VT"}
print(type_dict.values())
def token(type):
    ddata = pd.read_csv("data/ddata.txt", encoding="euc-kr", sep="\t", names=["num", "speaker", "contents", "tags"])
    mdata = pd.read_csv("data/mdata.txt", encoding="euc-kr", sep="\t", names=["id", "part", "category", "exptype", "exp"])
    dial_num = 0

    for i in range(1, len(ddata)):
        if ddata.loc[i, "num"] == 0:
            dial_num += 1
    c_mdata = pd.DataFrame({"id": [0], "exp": [None]})
    c_mdata.drop(0, axis=0, inplace=True)
    c_mdata
    id = ""
    idx = 0
    for i in range(len(mdata)):
        if id == mdata.loc[i, "id"]:
            c_mdata.loc[c_mdata.index[(c_mdata["id"] == id)].tolist().pop(), "exp"] = c_mdata.loc[c_mdata.index[(
                        c_mdata["id"] == id)].tolist().pop(), "exp"] + " " + mdata.loc[i, "exp"]

        else:
            id = mdata.loc[i, "id"]
            c_mdata.loc[i, "id"] = id
            c_mdata.loc[i, "exp"] = mdata.loc[i, "exp"]
    for i in range(len(c_mdata)):  # id 뒤에 공백 없애기
        c_mdata.iloc[i].id = c_mdata.iloc[i].id.strip()

    print(c_mdata.iloc[0].id)
    c_mdata.reset_index(drop=True, inplace=True)
    ddata.drop(ddata.index[ddata.tags == "INTRO"].tolist(), axis=0, inplace=True)
    ddata.reset_index(drop=True, inplace=True)
    ddata.fillna("", inplace=True)
    c_data = pd.DataFrame({"contents": [None], "id": ["xx-xxx"]})
    c_data.drop(0, axis=0, inplace=True)
    dial = ddata.loc[0, "contents"].strip()
    ac = ""
    for i in range(1, len(ddata)):
        cur_sen = ddata.loc[i, "contents"].strip()

        if ddata.loc[i, "num"] > ddata.loc[i - 1, "num"]:  # 이전 대화셋에 포함된 문장인지 확인

            if ddata.loc[i, "speaker"] == "<AC>":  # case 1. 의상 아이디가 나온 경우
                if len(ac) > 1:
                    ac = ac + " " + cur_sen  # 의상 아이디 저장
                else:
                    ac = cur_sen

            elif "USER_SUCCESS" == ddata.loc[i, "tags"]:  # case 2. USER_SUCCESS 태그가 나온 경우
                if 0 < len(ac) < 7:  # 의상을 하나만 추천한 경우
                    dial = dial + " " + ac + " " + cur_sen
                    c_data.loc[i, "contents"] = dial + " " + c_mdata.loc[c_mdata.id == ac].exp.values[0]
                    c_data.loc[i, "id"] = ac
                    ac = ""

                elif len(ac) == 0:
                    dial = dial + " " + cur_sen

                else:  # 의상을 한 번에 여러 개 추천한 경우
                    idx_cnt = 0
                    for one_ac in ac.split():
                        c_data.loc[i + idx_cnt, "contents"] = dial + " " + one_ac + " " + cur_sen + " " + \
                                                              c_mdata.loc[c_mdata.id == one_ac].exp.values[0]
                        c_data.loc[i + idx_cnt, "id"] = one_ac
                        idx_cnt += 1
                    dial = dial + " " + ac + " " + cur_sen
                    ac = ""

            elif "USER_FAIL" in ddata.loc[i, "tags"]:
                dial = dial + " " + ac + " " + cur_sen
                ac = ""

            else:  # case 3. 그냥 대화문인 경우
                dial = dial + " " + cur_sen

        else:  # 새로운 대화셋 시작이면 초기화
            dial = cur_sen
            ac = ""
    c_data.reset_index(drop=True, inplace=True)
    jk_data = pd.DataFrame({"contents": [None], "id": ["xx-xxx"]})
    jk_data.drop(0, axis=0, inplace=True)
    for i in range(len(c_data)):
        if type in c_data["id"][i][0:2]:
            jk_data = jk_data.append(c_data.iloc[i])

    jk_data.reset_index(drop=True, inplace=True)
    jk_data["contents"] = jk_data["contents"].str.replace("[^A-Za-z0-9ㄱ-ㅎㅏ-ㅣ가-힣 ]", "")
    jk_data["id"] = jk_data["id"].str.replace("[^0-9]", "")
    temp = jk_data["id"].tolist()
    type_count = dict(Counter(temp))
    del_list = []
    for i in type_count.keys():
        if type_count.get(i) == 1:
            del_list.append(i)
    for i in del_list:
        jk_data = jk_data[jk_data["id"] != i]
    jk_data.reset_index(drop=True, inplace=True)
    mecab = Mecab()
    x_data = []
    for i in range(len(jk_data)):
        x_data.append(mecab.morphs(jk_data["contents"][i]))
    stopwords = []
    with open('./data/불용어사전.txt', mode='rt', encoding='utf-8') as f:
        for word in f.readlines():
            word = word.strip()
            stopwords.append(word)
    X_data = []
    for sentence in x_data:  # 문장 하나 가져오고 단어 단위로 쪼개기
        temp_X = [word for word in sentence if not word in stopwords]  # 불용어 제거
        X_data.append(temp_X)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_data)
    threshold = 6
    total_cnt = len(tokenizer.word_index)  # 단어의 수
    rare_cnt = 0  # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
    total_freq = 0  # 훈련 데이터의 전체 단어 빈도수 총 합
    rare_freq = 0  # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합

    # 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.
    for key, value in tokenizer.word_counts.items():
        total_freq = total_freq + value

        # 단어의 등장 빈도수가 threshold보다 작으면
        if (value < threshold):
            rare_cnt = rare_cnt + 1
            rare_freq = rare_freq + value
    vocab_size = total_cnt - rare_cnt + 1
    tokenizer = Tokenizer(num_words=vocab_size - 1)
    tokenizer.fit_on_texts(X_data)
    # saving
    with open('{}_tokenizer'.format(type), 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
for type in type_dict.values():
    token(type)