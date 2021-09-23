#-*- coding:utf-8 -*-
import io
import argparse
from gensim.models import Word2Vec
from gensim.models import FastText
from tqdm import tqdm
# from embedding.models.word_eval import WordEmbeddingEvaluator


def get_text(args):
    dlg_list = []
    with io.open(args.DDATA_PATH, encoding='euc-kr') as f:
        i=0
        for a in f:
            b = a.strip().split('\t')
            if len(b) > 2 and b[1] != '<AC>':
                dlg_list.append(b[2])
    return dlg_list

def dlg_word2vec(filename,outputname,windows=5):
    courps = [sent.strip().split(" ") for sent in open(filename, 'r').readlines()]
    model = Word2Vec(courps, vector_size=100, workers=16, sg=1, window=windows,min_count=2)
    model.save(outputname)

def text_fasttext(filename,outputname,windows=5):
    courps = [sent.strip().split(" ") for sent in open(filename, 'r').readlines()]
    model = FastText(courps, vector_size=100, workers=16, sg=1, window=windows,min_count=2)
    model.save(outputname)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # p = argparse.ArgumentParser()
    # p.add_argument("-dpath", "--DDATA_PATH", default="dlg_text3.txt")
    # args = p.parse_args()


    # dlg_word2vec("data/textandembedding/allmorptok_emmbedingtextfile.txt","data/textandembedding/allmorptok_emmbedingtext_130_word2vec")
    # dlg_word2vec("data/textandembedding/dlgmorptok_emmbedingtextfile.txt","data/textandembedding/dlgmorptok_emmbedingtext_130_word2vec")
    # dlg_word2vec("data/textandembedding/wordtok_emmbedingtextfile.txt","data/textandembedding/wordtok_emmbedingtext_130_word2vec")

    # dlg_word2vec("dlg_text.txt","dlg_v3_word2vec",2)
    # model = WordEmbeddingEvaluator("dlg_text_word2vec",method="word2vec",dim=100, tokenizer_name="mecab")
    # model.most_similar("학교",topn=5)

    ## 단어 유사도 판단
    # model = Word2Vec.load('data/textandembedding/allmorptok_emmbedingtext_130_word2vec',130)
   #print(model.wv.most_similar("회사",topn=10))
    model1 = Word2Vec.load('data/textandembedding/allmorptok_emmbedingtext_130_word2vec')
    print(model1.wv.most_similar('치마',topn=10))

    # print(model.wv[u"화사"])

    ## fasttext
    # text_fasttext("data/textandembedding/allmorptok_emmbedingtextfile.txt","data/textandembedding/allmorptok_emmbedingtext_130_fasttext")
    # text_fasttext("data/textandembedding/dlgmorptok_emmbedingtextfile.txt","data/textandembedding/dlgmorptok_emmbedingtext_130_fasttext")
    # text_fasttext("data/textandembedding/wordtok_emmbedingtextfile.txt","data/textandembedding/wordtok_emmbedingtext_130_fasttext")
    ## 유사도 판단
    model_fasttext=FastText.load('data/textandembedding/allmorptok_emmbedingtext_130_fasttext')
    print(model_fasttext.wv.most_similar("오지나리 다지나도갸지",topn=20))
    print(model_fasttext["치마"])






