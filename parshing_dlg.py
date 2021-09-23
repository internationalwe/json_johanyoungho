# -*- coding: utf-8 -*-
import argparse
import io
from konlpy.tag import *
import platform
import time
import re
from collections import Counter

def get_dlg_text_sentence(args):
    dlg_list = []
    count12=1
    with io.open(args.DDATA_PATH, encoding='euc-kr') as f:
        temp = []
        con_temp=[]
        dlg_temp=[]
        previous_index = 0
        previous_ac_index = 0
        for a in f:
            if previous_index > int(a.strip().split('\t')[0].strip()): #다음 대화문으로 넘어갔을때 전 대화문것을 파싱
                count12 = count12+1
                max_index = 0 # 대화문 문장 길이 체크
                for con in con_temp: # 대화문 문장 길이 체크 포문
                    max_index = int(con.strip().split('\t')[0].strip())
                for con in con_temp: # 파싱하는 for문
                    b = con.strip().split('\t') # 한 문장을 스플릿
                    if len(b) > 3:
                        if "CONFIRM_SHOW" in b[3]:
                            break
                    if b[1] == '<AC>' and len(b[2].split(" ")) == 1: #AC가 나오고 AC가 하나일때 인경우
                        AC = "" #AC ID를 저장하는 변수
                        check = True # user sucess인지 faill인지 체크하는 변수
                        i = 0 # 그냥 변수
                        while True: # ac부터 문장 끝날때까지 USER찾고 있으면 success인지 fail인지 확인 안나오면 그냥 success
                            if len(con_temp[int(b[0].strip()) + i].strip().split('\t')) > 3 : # 한문장에 정보가 4개일때 즉 intro, exp, user가 있는 설명 문장인지 확인
                                if 'SUCCESS' in con_temp[int(b[0].strip()) + i].strip().split('\t')[3] or 'FAIL' in con_temp[int(b[0].strip()) + i].strip().split('\t')[3]: # 문장 설명에 USER가 들어있는지 확인
                                    if "SUCCESS" in con_temp[int(b[0].strip()) + i].strip().split('\t')[3] :
                                        check = True
                                        break
                                    if "FAIL" in con_temp[int(b[0].strip()) + i].strip().split('\t')[3] :
                                        check = False
                                        break
                            if int(con_temp[int(b[0].strip()) + i].strip().split('\t')[0]) == max_index: # 문장끝에 가면 그냥 빠져나옴
                                break
                            i = i + 1
                        if len (con_temp[int(b[0]) + 1].strip().split('\t')) > 3:
                            if con_temp[int(b[0]) + 1].strip().split('\t')[3][:3] == "EXP": # 다음문장에 EXP나오면 EXP를 dlg에 추가
                                con_dlg = dlg_temp[:int(b[0].strip())]+[dlg_temp[int(b[0].strip()) + 1]]
                                con_dlg = [sen for sen in con_dlg if sen != '\0']
                                con_dlg = [re.sub("!|\?|\.|,", "", str(sen)) for sen in con_dlg]
                                con_dlg = " ".join(con_dlg)
                                AC = b[2].strip()
                                if check: # success면 1 fail이면 0
                                    sentence = [con_dlg] + [AC] + ['1']
                                elif not check:
                                    sentence = [con_dlg] + [AC] + ['0']
                            else: # exp 없을 떄 그냥 앞에 문장만 dlg에
                                con_dlg = dlg_temp[:int(b[0].strip())]
                                con_dlg = [sen for sen in con_dlg if sen != '\0']
                                con_dlg = [re.sub("!|\?|\.|,", "", str(sen)) for sen in con_dlg]
                                con_dlg = " ".join(con_dlg)
                                AC = b[2].strip()
                                if check:
                                    sentence = [con_dlg] + [AC] + ['1']
                                elif not check:
                                    sentence = [con_dlg] + [AC] + ['0']
                        else:
                            con_dlg = dlg_temp[:int(b[0].strip())]
                            con_dlg = [sen for sen in con_dlg if sen != '\0']
                            con_dlg = [re.sub("!|\?|\.|,", "", str(sen)) for sen in con_dlg]
                            con_dlg = " ".join(con_dlg)
                            AC = b[2].strip()
                            if check:
                                sentence = [con_dlg] + [AC] + ['1']
                            elif not check:
                                sentence = [con_dlg] + [AC] + ['0']
                        dlg_list.append("\t".join(sentence))
                    elif b[1] == '<AC>' and len(b[2].split(" ")) > 1: #여러 옷이 있을때
                        AC = ""
                        check = True
                        i = 0
                        while True: #위에랑 똑같이 sucess랑 fail 찾기
                            if len(con_temp[int(b[0].strip()) + i].strip().split('\t')) > 3:
                                if 'SUCCESS' in con_temp[int(b[0].strip()) + i].strip().split('\t')[3] or 'FAIL' in \
                                        con_temp[int(b[0].strip()) + i].strip().split('\t')[3]:  # 문장 설명에 USER가 들어있는지 확인
                                    if "SUCCESS" in con_temp[int(b[0].strip()) + i].strip().split('\t')[3]:
                                        check = True
                                        break
                                    if "FAIL" in con_temp[int(b[0].strip()) + i].strip().split('\t')[3]:
                                        check = False
                                        break
                            if int(con_temp[int(b[0].strip()) + i].strip().split('\t')[0]) == max_index:
                                break
                            i = i + 1
                        if len(con_temp[int(b[0]) + 1].strip().split('\t')) > 3:
                            if con_temp[int(b[0]) + 1].strip().split('\t')[3][:3] == "EXP":
                                con_dlg = dlg_temp[:int(b[0].strip())]+[dlg_temp[int(b[0].strip()) + 1]]
                                con_dlg = [sen for sen in con_dlg if sen != '\0']
                                con_dlg = [re.sub("!|\?|\.|,", "", str(sen)) for sen in con_dlg]
                                con_dlg = " ".join(con_dlg)
                                for ac in b[2].strip().split(' '):
                                    if check:
                                        sentence = [con_dlg] + [ac] + ['1']
                                        dlg_list.append("\t".join(sentence))
                                    elif not check:
                                        sentence = [con_dlg] + [ac] + ['0']
                                        dlg_list.append("\t".join(sentence))
                            else:
                                con_dlg = dlg_temp[:int(b[0].strip())]
                                con_dlg = [sen for sen in con_dlg if sen != '\0']
                                con_dlg = [re.sub("!|\?|\.|,", "", str(sen)) for sen in con_dlg]
                                con_dlg = " ".join(con_dlg)
                                for ac in b[2].strip().split(' '):
                                    if check:
                                        sentence = [con_dlg] + [ac] + ['1']
                                        dlg_list.append("\t".join(sentence))
                                    elif not check:
                                        sentence = [con_dlg] + [ac] + ['0']
                                        dlg_list.append("\t".join(sentence))
                        else:
                            con_dlg = dlg_temp[:int(b[0].strip())]
                            con_dlg = [sen for sen in con_dlg if sen != '\0']
                            con_dlg = [re.sub("!|\?|\.|,", "", str(sen)) for sen in con_dlg]
                            con_dlg = " ".join(con_dlg)
                            for ac in b[2].strip().split(' '):
                                if check:
                                    sentence = [con_dlg] + [ac] + ['1']
                                    dlg_list.append("\t".join(sentence))
                                elif not check:
                                    sentence = [con_dlg] + [ac] + ['0']
                                    dlg_list.append("\t".join(sentence))
                con_temp = []
                dlg_temp = []
            if len(a.strip().split('\t')) > 2:
                con_temp.append(a)
                dlg_temp.append(a.strip().split('\t')[2].strip() if a.strip().split('\t')[1] != '<AC>' else '\0')
            previous_index = int(a.strip().split('\t')[0].strip())
    print(count12)
    return dlg_list

def get_dlg_text(args):
    dlg_list = []
    with io.open(args.DDATA_PATH, encoding='euc-kr') as f:
        temp = []
        con_temp=[]
        dlg_temp=[]
        previous_index = 0
        previous_ac_index = 0
        for a in f:
            if previous_index > int(a.strip().split('\t')[0].strip()): #다음 대화문으로 넘어갔을때 전 대화문것을 파싱
                max_index = 0 # 대화문 문장 길이 체크
                for con in con_temp: # 대화문 문장 길이 체크 포문
                    max_index = int(con.strip().split('\t')[0].strip())
                for con in con_temp: # 파싱하는 for문
                    b = con.strip().split('\t') # 한 문장을 스플릿
                    if len(b) > 3:
                        if "CONFIRM_SHOW" in b[3]:
                            break
                    if b[1] == '<AC>' and len(b[2].split(" ")) == 1: #AC가 나오고 AC가 하나일때 인경우
                        AC = "" #AC ID를 저장하는 변수
                        check = True # user sucess인지 faill인지 체크하는 변수
                        i = 0 # 그냥 변수
                        while True: # ac부터 문장 끝날때까지 USER찾고 있으면 success인지 fail인지 확인 안나오면 그냥 success
                            if len(con_temp[int(b[0].strip()) + i].strip().split('\t')) > 3 : # 한문장에 정보가 4개일때 즉 intro, exp, user가 있는 설명 문장인지 확인
                                if 'SUCCESS' in con_temp[int(b[0].strip()) + i].strip().split('\t')[3] or 'FAIL' in \
                                        con_temp[int(b[0].strip()) + i].strip().split('\t')[3]:  # 문장 설명에 USER가 들어있는지 확인
                                    if "SUCCESS" in con_temp[int(b[0].strip()) + i].strip().split('\t')[3]:
                                        check = True
                                        break
                                    if "FAIL" in con_temp[int(b[0].strip()) + i].strip().split('\t')[3]:
                                        check = False
                                        break
                            if int(con_temp[int(b[0].strip()) + i].strip().split('\t')[0]) == max_index: # 문장끝에 가면 그냥 빠져나옴
                                break
                            i = i + 1
                        if len (con_temp[int(b[0]) + 1].strip().split('\t')) > 3:
                            if con_temp[int(b[0]) + 1].strip().split('\t')[3][:3] == "EXP": # 다음문장에 EXP나오면 EXP를 dlg에 추가
                                con_dlg = dlg_temp[:int(b[0].strip())]+[dlg_temp[int(b[0].strip()) + 1]]
                                con_dlg = [sen for sen in con_dlg if sen != '\0']
                                con_dlg = [re.sub("!|\?|\.|,", "", str(sen)) for sen in con_dlg]
                                con_dlg = " ".join(con_dlg)
                                AC = b[2].strip()
                                if check: # success면 1 fail이면 0
                                    sentence = [con_dlg] + [AC] + ['1']
                                elif not check:
                                    sentence = [con_dlg] + [AC] + ['0']
                            else: # exp 없을 떄 그냥 앞에 문장만 dlg에
                                con_dlg = dlg_temp[:int(b[0].strip())]
                                con_dlg = [sen for sen in con_dlg if sen != '\0']
                                con_dlg = [re.sub("!|\?|\.|,", "", str(sen)) for sen in con_dlg]
                                con_dlg = " ".join(con_dlg)
                                AC = b[2].strip()
                                if check:
                                    sentence = [con_dlg] + [AC] + ['1']
                                elif not check:
                                    sentence = [con_dlg] + [AC] + ['0']
                        else:
                            con_dlg = dlg_temp[:int(b[0].strip())]
                            con_dlg = [sen for sen in con_dlg if sen != '\0']
                            con_dlg = [re.sub("!|\?|\.|,", "", str(sen)) for sen in con_dlg]
                            con_dlg = " ".join(con_dlg)
                            AC = b[2].strip()
                            if check:
                                sentence = [con_dlg] + [AC] + ['1']
                            elif not check:
                                sentence = [con_dlg] + [AC] + ['0']
                        dlg_list.append(sentence)
                    elif b[1] == '<AC>' and len(b[2].split(" ")) > 1: #여러 옷이 있을때
                        AC = ""
                        check = True
                        i = 0
                        while True: #위에랑 똑같이 sucess랑 fail 찾기
                            if len(con_temp[int(b[0].strip()) + i].strip().split('\t')) > 3:
                                if 'SUCCESS' in con_temp[int(b[0].strip()) + i].strip().split('\t')[3] or 'FAIL' in \
                                        con_temp[int(b[0].strip()) + i].strip().split('\t')[3]:  # 문장 설명에 USER가 들어있는지 확인
                                    if "SUCCESS" in con_temp[int(b[0].strip()) + i].strip().split('\t')[3]:
                                        check = True
                                        break
                                    if "FAIL" in con_temp[int(b[0].strip()) + i].strip().split('\t')[3]:
                                        check = False
                                        break
                            if int(con_temp[int(b[0].strip()) + i].strip().split('\t')[0]) == max_index:
                                break
                            i = i + 1
                        if len(con_temp[int(b[0]) + 1].strip().split('\t')) > 3:
                            if con_temp[int(b[0]) + 1].strip().split('\t')[3][:3] == "EXP":
                                con_dlg = dlg_temp[:int(b[0].strip())]+[dlg_temp[int(b[0].strip()) + 1]]
                                con_dlg = [sen for sen in con_dlg if sen != '\0']
                                con_dlg = [re.sub("!|\?|\.|,", "", str(sen)) for sen in con_dlg]
                                con_dlg = " ".join(con_dlg)
                                for ac in b[2].strip().split(' '):
                                    if check:
                                        sentence = [con_dlg] + [ac] + ['1']
                                        dlg_list.append(sentence)
                                    elif not check:
                                        sentence = [con_dlg] + [ac] + ['0']
                                        dlg_list.append(sentence)
                            else:
                                con_dlg = dlg_temp[:int(b[0].strip())]
                                con_dlg = [sen for sen in con_dlg if sen != '\0']
                                con_dlg = [re.sub("!|\?|\.|,", "", str(sen)) for sen in con_dlg]
                                con_dlg = " ".join(con_dlg)
                                for ac in b[2].strip().split(' '):
                                    if check:
                                        sentence = [con_dlg] + [ac] + ['1']
                                        dlg_list.append(sentence)
                                    elif not check:
                                        sentence = [con_dlg] + [ac] + ['0']
                                        dlg_list.append(sentence)
                        else:
                            con_dlg = dlg_temp[:int(b[0].strip())]
                            con_dlg = [sen for sen in con_dlg if sen != '\0']
                            con_dlg = [re.sub("!|\?|\.|,", "", str(sen)) for sen in con_dlg]
                            con_dlg = " ".join(con_dlg)
                            for ac in b[2].strip().split(' '):
                                if check:
                                    sentence = [con_dlg] + [ac] + ['1']
                                    dlg_list.append(sentence)
                                elif not check:
                                    sentence = [con_dlg] + [ac] + ['0']
                                    dlg_list.append(sentence)
                con_temp = []
                dlg_temp = []
            if len(a.strip().split('\t')) > 2:
                con_temp.append(a)
                dlg_temp.append(a.strip().split('\t')[2].strip() if a.strip().split('\t')[1] != '<AC>' else '\0')
            previous_index = int(a.strip().split('\t')[0].strip())
    return dlg_list

def dicmmdata(args):
    with io.open(args.MDATA_PATH, encoding='euc-kr') as md:
        pre_cloth = ""
        cloth_dic = {}
        for a in md:
            key = a.strip().split("\t")[0].strip()
            value = a.strip().split("\t")[4].strip()
            value = "".join(letter for letter in value if letter !=  "." or letter !=  "," or letter !=  "!" or letter !=  "?"or letter !=  "/")
            if key != pre_cloth:
                cloth_dic[key]=[value]
            else:
                cloth_dic[key].append(value)
            pre_cloth = key.strip()
    return cloth_dic


def make_wordtok_embbedingtxtfile(dlg_list,cloth_dic):
    with io.open("data/textandembedding/wordtok_emmbedingtextfile.txt", encoding='euc-kr', mode='w') as f:
        for temp in dlg_list:
            f.write(temp[0])
            f.write(' ')
            f.write(" ".join(cloth_dic[temp[1]]))
            f.write("\n")

def make_wordtok_inputtxtfile(dlg_list,cloth_dic):
    with io.open("data/textandembedding/wordtok_inputtextfile.txt", encoding='euc-kr', mode='w') as f:
        for temp in dlg_list:
            f.write(temp[0])
            f.write(' ')
            f.write(" ".join(cloth_dic[temp[1]]))
            f.write("\t")
            f.write(temp[2])
            f.write("\n")

def make_dlgmorptok_embbedingtxtfile(dlg_list,cloth_dic):
    with io.open("data/textandembedding/dlgmorptok_emmbedingtextfile.txt", encoding='euc-kr', mode='w') as f:
        for temp in dlg_list:
            tokdlg = mecab.morphs(temp[0])
            f.write(" ".join(tokdlg))
            f.write(' ')
            f.write(" ".join(cloth_dic[temp[1]]))
            f.write("\n")

def make_dlgmorptok_inputtxtfile(dlg_list,cloth_dic):
    with io.open("data/textandembedding/dlgmorptok_inputtextfile.txt", encoding='euc-kr', mode='w') as f:
        for temp in dlg_list:
            tokdlg = mecab.morphs(temp[0])
            f.write(" ".join(tokdlg))
            f.write(' ')
            f.write(" ".join(cloth_dic[temp[1]]))
            f.write("\t")
            f.write(temp[2])
            f.write("\n")

def make_allmorptok_embbedingtxtfile(dlg_list,cloth_dic):
    with io.open("data/textandembedding/allmorptok_emmbedingtextfile.txt", encoding='euc-kr', mode='w') as f:
        for temp in dlg_list:
            tokdlg = mecab.morphs(temp[0])
            f.write(" ".join(tokdlg))
            f.write(' ')
            tokmdata = mecab.morphs(" ".join(cloth_dic[temp[1]]))
            f.write(" ".join(tokmdata))
            f.write("\n")

def make_allmorptok_inputtxtfile(dlg_list,cloth_dic):
    with io.open("data/textandembedding/allmorptok_inputtextfile.txt", encoding='euc-kr', mode='w') as f:
        for temp in dlg_list:
            tokdlg = mecab.morphs(temp[0])
            f.write(" ".join(tokdlg))
            f.write(' ')
            tokmdata = mecab.morphs(" ".join(cloth_dic[temp[1]]))
            f.write(" ".join(tokmdata))
            f.write("\t")
            f.write(temp[2])
            f.write("\n")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument("-dpath", "--DDATA_PATH", default="Fashion-How/data/ddata.txt")
    p.add_argument("-mpath", "--MDATA_PATH", default="Fashion-How/data/mdata.txt")
    args = p.parse_args()

    mecab = Mecab('C:\mecab\mecab-ko-dic')

    dlg_list = get_dlg_text(args)
    dlg_list2 = get_dlg_text_sentence(args)
    true = 0
    false = 0
    for dlg in dlg_list2:
        if int(dlg.strip().split("\t")[2]) == 1:
            true = true + 1
        elif int(dlg.strip().split("\t")[2]) == 0:
            false = false + 1
    count_dlg = []
    for dlg in dlg_list:
        if dlg[2] == "1":
            count_dlg.append(dlg[1])
    counter = Counter(count_dlg)
    print(counter)
    freq_count = Counter(counter.values())
    print(freq_count)

    cloth_dic = dicmmdata(args)
    make_wordtok_inputtxtfile(dlg_list,cloth_dic)
    make_wordtok_embbedingtxtfile(dlg_list,cloth_dic)
    make_dlgmorptok_inputtxtfile(dlg_list,cloth_dic)
    make_dlgmorptok_embbedingtxtfile(dlg_list,cloth_dic)
    make_allmorptok_inputtxtfile(dlg_list, cloth_dic)
    make_allmorptok_embbedingtxtfile(dlg_list, cloth_dic)
    print(cloth_dic["CT-104"])
    print(len(dlg_list2))
    print("True : " + str(true))
    print("False : " + str(false))
    # print("Hello")