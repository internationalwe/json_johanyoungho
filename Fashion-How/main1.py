import argparse
import tensorflow as tf
import torch

from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np
import random
import time
import datetime
import io
from konlpy.tag import *

import os
cores = os.cpu_count()
torch.set_num_threads(cores)

def str2bool(v):
    """
    function: convert into bool type(True or False)
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_udevice():
    """
    function: get usable devices(CPU and GPU)
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        num_gpu = torch.cuda.device_count()
    else:    
        device = torch.device('cpu')
    print('Using device: {}'.format(device))
    if torch.cuda.is_available():
        print('# of GPU: {}'.format(num_gpu))
    device = torch.device('cpu')
    return device

parser = argparse.ArgumentParser(description='AI Fashion Coordinator.')

parser.add_argument('--mode', type=str, 
                    default='pred',
                    help='training or eval or test mode')
parser.add_argument('--in_file_trn_dialog', type=str, 
                    default='./data/ddata.wst.txt', 
                    help='training dialog DB')
parser.add_argument('--in_file_tst_dialog', type=str, 
                    default='./data/ac_eval_t1.wst.dev', 
                    help='test dialog DB')
parser.add_argument('--in_file_fashion', type=str, 
                    default='./data/mdata.wst.txt', 
                    help='fashion item metadata')
parser.add_argument('--in_file_img_feats', type=str, 
                    default='./data/extracted_feat.json', 
                    help='fashion item image features')
parser.add_argument('--model_path', type=str, 
                    default='./gAIa_model', 
                    help='path to save/read model')
parser.add_argument('--model_file', type=str, 
                    default="bert-4.pt",
                    help='model file name')
parser.add_argument('--eval_node', type=str, 
                    default='[6000,6000,6000,200][2000,2000]', 
                    help='nodes of evaluation network')
parser.add_argument('--subWordEmb_path', type=str, 
                    default='./sstm_v0p5_deploy/sstm_v4p49_np_final_n36134_d128_r_eng_upper.dat', 
                    help='path of subword embedding')
parser.add_argument('--learning_rate', type=float,
                    default=0.0001, 
                    help='learning rate')
parser.add_argument('--max_grad_norm', type=float,
                    default=40.0, 
                    help='clip gradients to this norm')
parser.add_argument('--zero_prob', type=float,
                    default=0.0, 
                    help='dropout prob.')
parser.add_argument('--corr_thres', type=float,
                    default=0.7, 
                    help='correlation threshold')
parser.add_argument('--batch_size', type=int,
                    default=100,   
                    help='batch size for training')
parser.add_argument('--epochs', type=int,
                    default=10,   
                    help='epochs to training')
parser.add_argument('--save_freq', type=int,
                    default=2,   
                    help='evaluate and save results per # epochs')
parser.add_argument('--hops', type=int,
                    default=3,   
                    help='number of hops in the MemN2N')
parser.add_argument('--mem_size', type=int,
                    default=16,   
                    help='memory size for the MemN2N')
parser.add_argument('--key_size', type=int,
                    default=300,   
                    help='memory size for the MemN2N')
parser.add_argument('--permutation_iteration', type=int,
                    default=3,   
                    help='# of permutation iteration')
parser.add_argument('--evaluation_iteration', type=int,
                    default=10,   
                    help='# of test iteration')
parser.add_argument('--num_augmentation', type=int,
                    default=3,   
                    help='# of data augmentation')
parser.add_argument('--use_batch_norm', type=str2bool, 
                    default=False, 
                    help='use batch normalization')
parser.add_argument('--use_dropout', type=str2bool, 
                    default=False, 
                    help='use dropout')
parser.add_argument('--use_multimodal', type=str2bool,
                    default=True,
                    help='use multimodal input')

args = parser.parse_args()



def dicmmdata(filepath):
    with io.open(filepath, encoding='euc-kr') as md:
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
def convert_input_data(sentences):

    tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
    MAX_LEN = 512
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    text_len = len(input_ids)
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
    attention_masks = []
    for seq in input_ids:
        seq_mask = [float(i>0) for i in seq]
        attention_masks.append(seq_mask)
    inputs = torch.tensor(input_ids)
    masks = torch.tensor(attention_masks)

    return inputs, masks

def test_sentences(sentences):
    model.eval()

    inputs, masks = convert_input_data(sentences)
    b_input_ids = inputs.to(device)
    b_input_mask = masks.to(device)
    with torch.no_grad():
        outputs = model(b_input_ids,
                        token_type_ids=None,
                        attention_mask=b_input_mask)
    logits = outputs[0]
    logits = logits.detach().cpu().numpy()

    return logits

def changekau(order):
    if order == [2,1,0]:
        return 5
    if order == [2,0,1]:
        return 4
    if order == [1,2,0]:
        return 3
    if order == [1,0,2]:
        return 2
    if order == [0,2,1]:
        return 1
    if order == [0,1,2]:
        return 0

if __name__ == '__main__':
    dlg_list = []
    with io.open(args.in_file_tst_dialog, encoding='euc-kr') as f:
        for a in f:
            b = a.strip().split('\t')
            dlg_list.append(b)
    con_list = []
    rnk1_list = []
    rnk2_list = []
    rnk3_list = []
    con_dlg_temp = []
    for dlg in dlg_list:
        if ';' in dlg[0] and dlg[0].strip() != '; 0':
            con_list.append(con_dlg_temp)
            con_dlg_temp = []
        if dlg[0] == 'CO' or dlg[0] == 'US':
            con_dlg_temp.append(dlg[1])
        elif dlg[0] == 'R1':
            rnk1_list.append(dlg[1].strip().split())
        elif dlg[0] == 'R2':
            rnk2_list.append(dlg[1].strip().split())
        elif dlg[0] == 'R3':
            rnk3_list.append(dlg[1].strip().split())
    seq_con_list = [" ".join(s) for s in con_list]
    for idx, sen in enumerate(seq_con_list):
        seq_con_list[idx] = "".join(s for s in sen if s != "." and s != ",")
    cloth_dic = dicmmdata(args.in_file_fashion)
    for idx, li in enumerate(rnk1_list):
        for jdx, cl in enumerate(li):
            rnk1_list[idx][jdx] = seq_con_list[idx] + " " + " ".join(cloth_dic[cl.strip()])
    for idx, li in enumerate(rnk2_list):
        for jdx, cl in enumerate(li):
            rnk2_list[idx][jdx] = seq_con_list[idx] + " " + " ".join(cloth_dic[cl.strip()])
    for idx, li in enumerate(rnk3_list):
        for jdx, cl in enumerate(li):
            rnk3_list[idx][jdx] = seq_con_list[idx] + " " + " ".join(cloth_dic[cl.strip()])
    mecab = Mecab('C:\mecab\mecab-ko-dic')
    for idx, le in enumerate(rnk1_list):
        for jdx, _ in enumerate(le):
            word1 = mecab.morphs(rnk1_list[idx][jdx])
            word2 = mecab.morphs(rnk2_list[idx][jdx])
            word3 = mecab.morphs(rnk3_list[idx][jdx])
            rnk1_list[idx][jdx] = " ".join(word1)
            rnk2_list[idx][jdx] = " ".join(word2)
            rnk3_list[idx][jdx] = " ".join(word3)
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    device = torch.device("cpu")
    model = torch.load("{}/{}".format(args.model_path,args.model_file),map_location=torch.device('cpu'))
    model.eval()
    predict_rank = []
    for idx, le in enumerate(rnk1_list):
        pred1 = 0.0
        pred2 = 0.0
        pred3 = 0.0
        for jdx in range(len(le)):
            print(jdx)
            pred1 = pred1 + test_sentences([rnk1_list[idx][jdx]])[0][1] - test_sentences([rnk1_list[idx][jdx]])[0][0]
            pred2 = pred2 + test_sentences([rnk2_list[idx][jdx]])[0][1] - test_sentences([rnk1_list[idx][jdx]])[0][0]
            pred3 = pred3 + test_sentences([rnk3_list[idx][jdx]])[0][1] - test_sentences([rnk1_list[idx][jdx]])[0][0]
        predict_rank.append([pred1, pred2, pred3])
    print(predict_rank)
    temp_df = pd.DataFrame
    order_list = []
    for idx, dwk in enumerate(predict_rank):
        temp_list = []
        del temp_df
        for jdx in range(3):
            temp_list.append([predict_rank[idx][jdx], jdx])
        print(temp_list)
        temp_df = pd.DataFrame(temp_list, columns=["value", "idx"])
        order_list.append(list(temp_df.sort_values(by=['value'], axis=0)["idx"]))
    kau_list = []
    for aklw in order_list:
        kau_list.append(changekau(aklw))
    kau_list = np.array(kau_list)
    np.savetxt("./prediction.csv", kau_list.astype(int), encoding='utf8', fmt='%d')
