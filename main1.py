import argparse
import io
from konlpy.tag import *
import platform
import time


def get_text(args):
    dlg_list = []
    with io.open(args.DDATA_PATH, encoding='euc-kr') as f:
        i=0
        for a in f:
            b = a.strip().split('\t')
            if len(b) > 2 and b[1] != '<AC>':
                dlg_list.append(b[2])
    return dlg_list

def get_make_mmdata(args):
    dlg_list = []
    with io.open(args.MDATA_PATH, encoding='euc-kr') as f:
        with io.open("data/textandembedding/焊包/mmdata_text.txt", encoding='euc-kr', mode='w')as md:
            i = 0
            previous_cloth="BL-001"
            for a in f:
                b = a.strip().split('\t')
                word = mecab.morphs(b[4])
                if previous_cloth == b[0]:
                    md.write(" ".join(str(i) for i in word if i != '.' and i != ',' and i != '?' and i != '!'))
                    md.write(' ')
                else:
                    md.write(" ".join(str(i) for i in word if i != '.' and i != ',' and i != '?' and i != '!'))
                    md.write('\n')
                dlg_list.append(b[4])
                previous_cloth = b[0]
    return dlg_list
def make_dlg_file():
    with io.open("data/textandembedding/焊包/dlg_text.txt", encoding='euc-kr', mode='w') as f:
        start = time.time()
        for a in dlg_text:
            b = mecab.morphs(a)
            f.write(" ".join(str(i) for i in b if i != '.' and i != ',' and i != '?' and i != '!'))
            f.write("\n")
def make_mmdata_file():
    with io.open("data/textandembedding/焊包/mmdata_text.txt", encoding='euc-kr', mode='w') as f:
        start = time.time()
        for a in mmdata_text:
            b = mecab.morphs(a)
            f.write(" ".join(str(i) for i in b if i != '.' and i != ',' and i != '?' and i != '!'))
            f.write("\n")

def cat_file():
    with open("data/textandembedding/焊包/cattext.txt", 'w') as f:
        with open("data/textandembedding/焊包/dlg_text.txt", 'r') as dlg:
            with open("data/textandembedding/焊包/mmdata_text.txt", 'r') as md:
                for a in dlg:
                    f.write(a)
                for a in md:
                    f.write(a)


if __name__ == '__main__':

    p = argparse.ArgumentParser()
    p.add_argument("-dpath", "--DDATA_PATH", default="Fashion-How/data/ddata.txt")
    p.add_argument("-mpath", "--MDATA_PATH", default="Fashion-How/data/mdata.txt")
    args = p.parse_args()
    # hannanum = Hannanum()
    # komoran = Komoran()
    mecab = Mecab('C:\mecab\mecab-ko-dic')
    # okt = Okt()
    # kkma = Kkma()

    # hannanum_dlglist =[]
    # komoran_dlglist =[]
    mecab_dlglist =[]
    # okt_dlglist =[]
    # kkma_dlglist =[]
    time_list = {}

    dlg_text = get_text(args)
    make_dlg_file()
    mmdata_text = get_make_mmdata(args)

    cat_file()

    # start = time.time()
    # for a in dlg_text:
    #     b = hannanum.morphs(a)
    #     hannanum_dlglist.append(b)
    # time_list['hannanum'] = (time.time() - start)
    # print(time_list)
    #
    # start = time.time()
    # for a in dlg_text:
    #     b = komoran.morphs(a)
    #     komoran_dlglist.append(b)
    # time_list['komoran'] = (time.time() - start)
    # print(time_list)
    #
    # with open("dlg_tex.txt", 'w') as f:
    #     start = time.time()
    #     for a in dlg_text:
    #         b = mecab.morphs(a)
    #         mecab_dlglist.append(b)
    #         for word in b:
    #             senten = ""
    #             if word != '.' and word != ',' and word != '?' and word != '!':
    #                 senten = senten + word
    #                 senten = senten + " "
    #         f.write(senten.strip())
    #         f.write("\n")




    #
    # start = time.time()
    # for a in dlg_text:
    #     b = okt.morphs(a)
    #     okt_dlglist.append(b)
    # time_list['okt'] = (time.time() - start)
    # print(time_list)
    #
    # start = time.time()
    # for a in dlg_text:
    #     b = kkma.morphs(a)
    #     kkma_dlglist.append(b)
    # time_list['kkma'] = (time.time() - start)
    # print(time_list)
    # print(time_list)