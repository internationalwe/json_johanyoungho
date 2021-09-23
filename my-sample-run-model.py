#-*- coding: euc-kr -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import Counter

import argparse
import numpy as np
import sys
import json
#import _pickle as pickle
import pickle
import io

import tensorflow as tf
import os


"""
2020.7.3 ETRI �������� ������ ���Ǽ�(eschung@etri.re.kr)
ç������ �򰡼� ������ ����
"""



class ACBotImportedModel:

    def __init__(self, ac_data, loc):
        """ �н� ����� import�ϰ�, �Է� ��� tensor�� ����
            tensorflow ���� �ʱ�ȭ
        """
        self.ac_data = ac_data
        self.loc = loc
        self.graph = tf.Graph()
        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.allow_growth = True
        self.session = tf.Session(graph=self.graph, config=sess_config)

        with self.graph.as_default():
            saver = tf.train.import_meta_graph(loc+".meta")
            saver.restore(self.session, loc)

            self.X_SP        = self.graph.get_tensor_by_name("X_SP:0")
            self.X_SP_LEN    = self.graph.get_tensor_by_name("X_SP_LEN:0")
            self.X_AC        = self.graph.get_tensor_by_name("X_AC:0")
            self.keep_probs  = self.graph.get_tensor_by_name("keep_prob:0")
            self.predictions = self.graph.get_tensor_by_name("predictions:0")

    def predict(self, sp, sp_len, ac):
        """ sp �� voca id�� ������ string�� batch
            sp_len�� �� string�� ���� ���� batch
            ac�� sp�� ���� �ǻ� ID ����Ʈ batch
        """

        feed_dict = { self.X_SP:sp, self.X_SP_LEN:sp_len, self.X_AC:ac, self.keep_probs:1.0 }
        pred = self.session.run([ self.predictions ], feed_dict=feed_dict)

        return pred[0]
      




class ACData:

    def __init__(self, args):
        """ �н� ��ȭ�°� �ǻ� ��Ÿ �����͸� �ε� ��, voca�� �ʱ�ȭ
            �򰡼��� �ε� voca id�� �̿��Ͽ� ���ڵ� ����
        """
        self.dlg_list = self.load_dlg_data( args.DDATA_PATH )
        self.mt_list  = self.load_mt_data( args.MDATA_PATH ) 
        self.enc_word_to_id = self.build_enc_voca()
        self.ac_dic_emb_inx, self.ac_dic_v_maxlen = self.build_ac_dic()
        self.eval_sp, self.eval_sp_len, self.eval_ac_rank = self.load_eval_data( args.EVAL_PATH )



    def load_dlg_data(self, dlg_text):
        """ ��ȭ�� ���� dlg_text�� �ε� 
        """
        dlg_list = []
        with io.open (dlg_text, encoding='euc-kr') as f:
            for a in f:
                b = a.strip().split('\t')
                dlg_list.append(b)
        return self.dlg_list_post_process(dlg_list)



    def dlg_list_post_process(self, dlg_list):
        """ dlg_list �� <AC>�� �ڵ� OTBS�������� �����Ѵ�. 
        """
        AC_RESET = ['None', 'None', 'None', 'None']

        def UpdateAC(preAC, currentAC):
            """ currentAC �� �ǻ�ID �ڵ� ����� OTBS������ �����ϰ�, 
                �ش� ���Կ� �Ҵ��Ͽ� preAC�� �����Ѵ�.  
            """
            otbs_inx = { 'CT':0, 'CD':0, 'VT':0, 'JK':0, 'JP':0, 
                         'KN':1, 'SW':1, 'SH':1, 'BL':1, 'T_KN':1, 
                         'SK':2, 'PT':2, 'OP':2, 'SE':3 }

            for a in currentAC.split():
                at = a.split('-')[0]
                if at in otbs_inx: 
                    if '_' in at: a = a[2:]
                    preAC[ otbs_inx[at] ] = a
                elif at == 'NONE':
                    if a[5] == 'O': preAC[0] = 'None'
                    elif a[5] == 'T': preAC[1] = 'None'
                    elif a[5] == 'B': preAC[2] = 'None'
                    elif a[5] == 'S': preAC[3] = 'None'
                else:
                    print ("error", a)
                    assert 0

        """ dlg_list�� <AC>�� otbs������������ �����Ѵ� """
        currentAC = AC_RESET[:]
        pre_i = 0
        for inx, a in enumerate(dlg_list):
            if int(a[0]) < pre_i:
                currentAC = AC_RESET[:]
            if a[1] == '<AC>':
                UpdateAC(currentAC, a[2])
                a[2] = " ".join(currentAC)

            pre_i = int(a[0])

        """ <CO> <AC> <US>  �� <AC> <CO> <US> �� ���� """
        for a in range(len(dlg_list)-1):
            if dlg_list[a][1] == '<CO>' and dlg_list[a+1][1] == '<AC>' and dlg_list[a+2][1] == '<US>':
                tmp = dlg_list[a]
                dlg_list[a] = dlg_list[a+1]
                dlg_list[a+1] = tmp
                tmp2 = dlg_list[a][0]
                dlg_list[a][0] = dlg_list[a+1][0]
                dlg_list[a+1][0] = tmp2

        return dlg_list



    def load_mt_data(self, mt_text):
        """ �ǻ� ��Ÿ ���� mt_text�� �ε� """
        mt_list  = [['None', '', '', '', '']]
        with io.open (mt_text, encoding='euc-kr') as f:
            for a in f:
                b = a.strip().split('\t')
                b = [ c.strip() for c in b ]
                mt_list.append(b)
        return mt_list





    def build_enc_voca(self, min_count=1):
        """ ���ڴ��� voca�� ���� : enc_word_to_id & enc_id_to_word
        """
        # �ǻ� ��Ÿ ������ �߰����� ����
        cloth_type = { 'CT':'��Ʈ', 'CD':'�����', 'VT':'����', 'JK':'����',     'JP':'�ĵ�',
                       'KN':'����', 'SW':'������', 'SH':'����', 'BL':'���콺', 'SK':'��ĿƮ', 
                       'PT':'����', 'OP':'���ǽ�', 'SE':'�Ź�' }

        for a in self.mt_list[1:]:
            ctype = a[2].strip()
            if cloth_type[ctype] not in a[4]:
                a[4] = a[4] + " " + cloth_type[ctype]

        enc_voca = []

        # �ǻ��Ÿ���� ����
        for a in self.mt_list[1:]:
            enc_voca.append(a[0])
            words = a[4].split()
            for b in words:
                enc_voca.append(b)

        # ��ȭ�� ����
        for a in self.dlg_list:
            words = a[2].split()
            for b in words: 
                enc_voca.append(b)

        enc_voca = Counter(enc_voca)

        new_enc_voca = []

        for key in enc_voca.keys():
            if enc_voca[key] >=  min_count:
                new_enc_voca.append(key)

        new_enc_voca   = ['<PAD>', '<UNK>', '<SS>'] + new_enc_voca
        enc_word_to_id = {word:i for i, word in enumerate(new_enc_voca)}
        enc_id_to_word = {i:word for i, word in enumerate(new_enc_voca)}

        print ('len(new_enc_voca)=', len(new_enc_voca))

        return enc_word_to_id







    def build_ac_dic(self):
        """ �ǻ� id�� �ε����� ���� 
        """
        ac_dic = {'None':[0,0]}

        for i, a in enumerate(self.mt_list):
            if a[0] in ac_dic:
                ac_dic[a[0]][1] = i 
            else:
                ac_dic[a[0]] = [i, i]

        ac_dic_key_list = ['None'] + [ a for a, b in ac_dic.items() if a != 'None' ] 
        ac_dic_emb_inx = { j:i for i, j in enumerate(ac_dic_key_list)} 
        ac_dic_v_maxlen = max([b[1]-b[0]+1 for a, b in ac_dic.items()])

        return ac_dic_emb_inx, ac_dic_v_maxlen






    def load_eval_data(self, fn):
        """  �� ������ ó��
        """

        def sp_enc(s):
            encoded_line = []
            for a in s.split():
                if a not in self.enc_word_to_id: encoded_line.append(self.enc_word_to_id['<UNK>'])
                else:                            encoded_line.append(self.enc_word_to_id[a])
            return encoded_line


        def get_ctx(sp_list, fixed_len = 10, max_len = 400):
            """  ��ȭ ���ؽ�Ʈ�� voca id�� ���ڵ��� string���� ��ȯ 
                 10�� ���� ����, 400�ܾ� ���� ����
            """
            ctx_sp, ctx_sp_len = [], []
            for b in sp_list:
                ctx_sp.append( b + [self.enc_word_to_id['<SS>']] )
                ctx_sp_len.append( len(b)+1 )

            ctx_sp[-1] = ctx_sp[-1][:-1]
            ctx_sp_len[-1] = ctx_sp_len[-1]-1

            if len(ctx_sp) > fixed_len:
                ctx_sp = ctx_sp[:fixed_len]
                ctx_sp_len = ctx_sp_len[:fixed_len]

            s_ret = sum(ctx_sp,[]) # [[1,2], [3,4]] -> [1,2,3,4]
            l_ret = sum(ctx_sp_len)
            
            if l_ret > max_len:
                l_ret = max_len
                s_ret = s_ret[:max_len]
            else:
                s_ret = s_ret + [self.enc_word_to_id['<PAD>']]*(max_len-l_ret)

            return np.array(s_ret), np.array(l_ret) 


        def get_ac(ac):
            """ �򰡼��� �ǻ� ID�� voca�� �ǻ� ID�� ��ȯ�Ͽ� OTBS 4���� �������� �Ҵ� 
            """
            otbs_inx = { 'CT':0, 'CD':0, 'VT':0, 'JK':0, 'JP':0, 'KN':1, 'SW':1, 'SH':1, 'BL':1, 
                         'SK':2, 'PT':2, 'OP':2, 'SE':3 }

            def get_ac_otbs(a):
                ac_otbs = ['None', 'None', 'None', 'None']
                for b in a.split():
                    at = b.split('-')[0]
                    if at in otbs_inx:
                        ac_otbs [ otbs_inx[at] ] = b
                    else:
                        print ("error", at)
                        print (a)
                        assert 0
                return ac_otbs

            ac_list = []

            for a in ac:
                b = [ self.ac_dic_emb_inx[c] for c in get_ac_otbs(a) ]
                ac_list.append(b)

            return np.array(ac_list)


        enc_sp, enc_sp_len = [], []
        rank_ac = []

        with io.open (fn, encoding='euc-kr') as f:
            sp, ac = [], []

            for a in f:

                if a[0] == ';' and len(sp) > 0 and len(ac) > 0:
                    assert len(ac) == 3
                    sp_list = [ sp_enc(b) for b in sp ]
                    _sp, _sp_len = get_ctx(sp_list)
                    enc_sp.append(_sp)
                    enc_sp_len.append(_sp_len)
                    rank_ac.append(get_ac(ac))
                    sp, ac = [], []

                elif a[0] == 'R':
                    ac.append(a.strip().split('\t')[1])

                elif a[0] == 'U' or a[0] == 'C':
                    sp.append(a.strip().split('\t')[1])

        return enc_sp, enc_sp_len, rank_ac








def ACBotTest( args, ac_data ):
    """ ���н��� ���� ����Ʈ�ϰ�, �򰡼��� �ϳ��� �ε��Ͽ�
        ����� ���ϰ�, weigted Kendall's tau�� �̿��Ͽ� �� ����
        �򰡼��� ������ ������ ( ��ȭ���ؽ�Ʈ s, [rank1 �ڵ� r1, rank2 �ڵ� r2, rank3 �ڵ� r3] )
        �������� ���ؽ�Ʈ�� �ߺ��Ͽ� 3���� ��ġ���ϰ�, ���� �ڵ���� �и��Ͽ� �ѹ��� 3���� ����
        ���Ѵ�.
             enc_sp_batch      = [s,s,s], 
             enc_sp_len_batch  = [len(s), len(s), len(s)], 
             enc_ac_batch      = [ r1, r2, r3 ]

             result = [ score(s, r1), score(s, r2), score (s, r3) ]

        result�� argsort�� �����Ͽ� rank�� ���Ѵ�. 
        
             rank(result) => [ 2, 1, 0 ]   # score�� ������ ���� �� (max=2, min=0)

        scipy�� weightedtau( ����rank����Ʈ, rank(result) )�� ���Ѵ�. default option ����.

            ���� rank ����Ʈ�� [2,1,0]���� ����

            wkt ( [2,1,0], [2,1,0] ) = 1.0
            wkt ( [2,1,0], [2,0,1] ) = 0.545454...
            wkt ( [2,1,0], [1,2,0] ) = 0.181818...
            wkt ( [2,1,0], [1,0,2] ) = -0.363636... 
            wkt ( [2,1,0], [0,2,1] ) = -0.363636...
            wkt ( [2,1,0], [0,l,2] ) = -0.999999...

            - rank(result)�� ���� �ߺ��� ������� �ʴ´�. ex [2,2,0], [0,0,1] �� 
            - ���� ���� ����� test set�� ����rank����Ʈ�� �����Ͽ� �����Ѵ�.  

        �ش� ���� ����� ����� ����
    """

    class_a = ACBotImportedModel( ac_data, args.IMPORTED_MODEL )

    rank_crr = 0.0
    final_result = []

    for ev_i in range( len( ac_data.eval_sp )):
        enc_sp_batch, enc_sp_len_batch, enc_ac_batch = [], [], []

        for ac_r in ac_data.eval_ac_rank[ev_i]:
            enc_sp_batch.append( ac_data.eval_sp[ev_i] )
            enc_sp_len_batch.append( ac_data.eval_sp_len[ev_i] )
            enc_ac_batch.append( ac_r )

        result = class_a.predict(enc_sp_batch, enc_sp_len_batch, enc_ac_batch)
        inx = list(np.array(result).argsort())
        r_pred = [0,0,0]
        r_pred[inx[0]] = 0
        r_pred[inx[1]] = 1
        r_pred[inx[2]] = 2

#r_pred = list(np.array(result).argsort())
#r_pred = [int(i) for i in r_pred] 
        final_result.append(r_pred)


    return final_result




output_path = "./predictions.txt"

def main():
    p = argparse.ArgumentParser()

    p.add_argument("-model", "--IMPORTED_MODEL", default="./model/model_do05-511")
    p.add_argument("-dpath", "--DDATA_PATH",     default="ddata.wst.txt.2020.6.23")
    p.add_argument("-mpath", "--MDATA_PATH",     default="mdata.wst.txt.2020.6.23")
    p.add_argument("-epath", "--EVAL_PATH",      default="ac_eval_t1.wst.dev")

    args = p.parse_args()

    ac_data = ACData( args )
    result = ACBotTest( args, ac_data )
    f = open(output_path, "w")
    for r in result:
       f.write("%d %d %d\n" % (r[0], r[1], r[2]))
    f.close()





if __name__ == '__main__':
    main()



