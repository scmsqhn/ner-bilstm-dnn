import sys
import gensim
import pymongo
import traceback
from dmp.gongan.ssc_dl_ner.data_utils import full_to_half
import numpy as np
import pandas as pd
import logging
import re
import pdb
import time
import os
import jieba
import collections
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sklearn.utils
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import json
from sklearn.model_selection import train_test_split
CURPATH = os.path.dirname(os.path.realpath(__file__))
PARPATH = os.path.dirname(CURPATH)
sys.path.append(PARPATH)
sys.path.append(CURPATH)
#from bilstm import addr_classify
#from bilstm import eval_bilstm
import datahelper
DEBUG =True
DATA = True
import gensim

def _path(filepath):
    CURPATH = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(CURPATH, filepath)

jieba.load_userdict(_path("all_addr_dict.txt"))

def logging_init(filename="logger.log"):
    filename = _path(filename)
    logger = logging.getLogger("bilstm_train.logger")
    logger.setLevel(level = logging.INFO)
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

lgr = logging_init()
pred_lgr = logging_init(filename="eval.log")

def _print(*l):
    logger = lgr
    if type(l) == str:
        logger.info(l)
    if type(l) == list:
        logger.info(str(l))
    if type(l) == tuple:
        logger.info(str(l))

def _print_pred(*l):
    logger = pred_lgr
    if type(l) == str:
        logger.info(l)
    if type(l) == list:
        logger.info(str(l))
    if type(l) == tuple:
        logger.info(str(l))

tran_prob = {'06': 0.00011000110001100011,\
 '11': 0.00016000160001600016,\
 '12': 0.02900029000290003,\
 '13': 0.24118241182411823,\
 '14': 0.00014000140001400014,\
 '21': 3.000030000300003e-05,\
 '22': 0.014580145801458014,\
 '23': 0.02895028950289503,\
 '24': 2.00000000200002e-05,\
 '31': 0.2108521085210852,\
 '34': 0.05918059180591806,\
 '35': 9.000090000900009e-05,\
 '37': 1.000010000100001e-05,\
 '41': 0.05886058860588606,\
 '44': 0.08012080120801209,\
 '45': 0.012880128801288013,\
 '47': 2.000020000200002e-05,\
 '50': 0.00011000110001100011,\
 '51': 5.000050000500005e-05,\
 '54': 0.0001000010000100001,\
 '56': 0.1128511285112851,\
 '61': 0.00025000250002500023,\
 '64': 0.012230122301223013,\
 '65': 0.08741087410874109,\
 '67': 0.013070130701307013,\
 '71': 0.0002800028000280003,\
 '74': 9.000090000900009e-05,\
 '75': 0.012730127301273013,\
 '77': 0.024640246402464025}

class Eval_Ner(object):

    def __init__(self):
        self.data_helper=datahelper.Data_Helper()
        self.init_eval_graph()
        self.tags = {'o':0,'b':1,'i':2}# words bg mid end / addrs bg mid end
        self.rev_tags = dict(zip(self.tags.values(), self.tags.keys()))

    def tag_map(self, pred_lst_1d):#[0:7]
        _ = list(pred_lst_1d)
        return self.rev_tags[_.index(max(_))]

    def init_eval_graph(self):
        tfconfig = tf.ConfigProto()
        tfconfig.gpu_options.allow_growth = True
        self.sess = tf.Session(config=tfconfig)
        self.sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(CURPATH)
        saver = tf.train.import_meta_graph(_path('bilstm.ckpt-1.meta'))
        saver.restore(self.sess, _path('bilstm.ckpt-1'))
        self.X_inputs=tf.get_collection("model.X_inputs")[0]
        self.y_inputs=tf.get_collection("model.y_inputs")[0]
        self.y_pred_meta=tf.get_collection("attention")[0]
        self.lr=tf.get_collection("lr")[0]
        self.batch_size=tf.get_collection("batch_size")[0]
        self.keep_prob=tf.get_collection('keep_prob')[0]
        self.FP = tf.get_collection('FP')[0]
        self.TN = tf.get_collection('TN')[0]
        self.FN = tf.get_collection('FN')[0]
        self.TP = tf.get_collection('TP')[0]
        self.Precision = tf.get_collection('Precision')[0]
        self.Recall = tf.get_collection('Recall')[0]
        self.Fscore = tf.get_collection('Fscore')[0]
        self.correct_prediction = tf.equal(tf.cast(tf.argmax(self.y_pred_meta, 1), tf.int32), tf.reshape(self.y_inputs, [-1]))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

    def judge(self, *para):
        paras = [thing for count,thing in enumerate(para)]
        try:
            paras = list(para)
            base = paras[0]
            for i in paras[1:]:
                if base == i:
                    continue
                return False
            return True
        except AssertionError:
            raise Exception("assert that what")
            return -1 
        except:
            if DEBUG:
                traceback.print_exc()
                return -2

    def data_prepare_txt(self, fname="./gz_jyaq.txt"):
        result = []
        f= open(fname)
        cont = f.read()
        lines= cont.split("\n")
        for line in lines:
            sent = list(jieba.cut(line))
            res = []
            if len(sent)>200:
                res.extend(sent[:200])
            else:
                res.extend(sent)
                for i in range(200-len(sent)):
                    res.extend([" "])
            print(len(res))
            assert len(res) == 200
            result.append(res)
            if len(result)%32==0 and len(result)>1:
                print(np.array(result).shape)
                yield result
                result = []

    def data_prepare(self):
        res = []
        datasrc = self.data_helper
        evalGen = datasrc.gen_train_data("eval")
        evalTaiyuanGen = datasrc.gen_train_data("evalTaiyuan")
        batchEval=datasrc.next_batch_eval(evalGen)
        batchEvalTaiyuan=datasrc.next_batch_eval(evalTaiyuanGen)
        res.append(batchEvalTaiyuan.__next__())
        res.append(batchEvalTaiyuan.__next__())
        res.append(batchEval.__next__())
        res.append(batchEval.__next__())
        return res

    def sent2WordsUnit(self, sent, num=200):
        words = list(jieba.cut(sent))
        ids = [self.data_helper.fromdct(word,flag=False) for word in words]
        tags = [0 for word in words]
        if len(ids) <num:
            lenSub = num - len(ids)
            ids.extend([self.data_helper.fromdct(" ", flag=False) for i in range(lenSub)])
            tags.extend([0 for i in range(lenSub)])
            # pdb.set_trace()
            return ids[:num], tags[:num]
        else:
            return ids[:num], tags[:num]

    def sent2BatchUnit(self,  sents, num=32):
        xbatch,ybatch = [],[]
        cnt=0
        for sent in sents:
            cnt+=1
            wordsTwoHundred, tagsTwohundred= self.sent2WordsUnit(sent)    
            xbatch.append(wordsTwoHundred)
            ybatch.append(tagsTwohundred)
            if cnt%num == 0:
                yield xbatch, ybatch
                xbatch, ybatch = [],[]
        if cnt%32>0:
            for i in range(num-cnt%32):
                p,q = self.sent2WordsUnit(" ")
                xbatch.append(p)
                ybatch.append(q)
            yield xbatch, ybatch

    def clrSet(self, setItem):
        if "" in setItem:
            setItem.remove("")
        return list(setItem)


    def predict_txt(self, text):
        res = []
        gen = self.sent2BatchUnit(text,32)
        while(1):
            X_in, y_in = -1,-1
            try:
                X_in, y_in = gen.__next__()
            except StopIteration:
                print("> data feed over")
                break
            X_batch, y_batch = np.array(X_in).reshape(32,200), np.array(y_in).reshape(32,200)
            feed_dict = {self.X_inputs:X_batch, self.y_inputs:y_batch, self.batch_size:32, self.keep_prob:1.0}
            print("x",X_batch)
            print("y",y_batch)
            fetches = [self.y_pred_meta]
            [yPredMeta] = self.sess.run(fetches, feed_dict=feed_dict) # the cost is the mean cost of one batch
            y_=np.argmax(yPredMeta.reshape(6400,3),1).reshape(32,200)
            for i,j in zip(X_in, y_):
                wordsStr = ""
                assert len(i) == len(j)
                assert len(i) == 200
                for m,n in zip(i,j):
                    #print(m,n)
                    #print(m,self.rev_tags[n])
                    wordsStr+="%s/%s "%(self.data_helper.dct.get(m), self.rev_tags[n])
                print(wordsStr)
                #pickTuple = re.split(".+?/[bi] ", wordsStr)
                pickTuple=re.split("[\u4e00-\u9fa5a-zA-Z0-9@\.]*/o",wordsStr)
                print(pickTuple)
                addrSet = set()
                for i in pickTuple:
                    ret = "".join(re.findall("([\u4e00-\u9fa5a-zA-Z0-9@\.]*)/[ib]",i))
                    addrSet.add(ret)
                res.append(self.clrSet(addrSet))
        return res

    def run_sent(self, sent):
        result, _, _= self.predict_sent(sent)
        for sent in result:
            _char_lst, _tags_pred_lst, _y_batch_lst = sent[0], sent[1], sent[2]
            _print(_char_lst, _tags_pred_lst, _y_batch_lst)
            assert len(_char_lst) == len(_tags_pred_lst)
            assert len(_char_lst) == len(_y_batch_lst)
            item_pred, item_base = "", ""
            for i,j,k in zip(_char_lst, _tags_pred_lst, _y_batch_lst):
                item_pred += "%s/%s "%(i, k)
                item_base += "%s/%s "%(i, j)
                item_pred = re.sub(" /v ","",item_pred)
                item_base = re.sub(" /v ","",item_base)
            basewords = list(re.findall("(./[ard](?:.*?)./[avrd] )", item_base))
            base_word_lst = []
            for word in basewords:
                _word = "".join(list(re.findall("(.)/. ",word)))
                base_word_lst.append(_word)
            predwords = list(re.findall("(./[ard](?:.*?)./[avrd] )", item_pred))
            pred_word_lst = []
            for word in predwords:
                _word = "".join(list(re.findall("(.)/. ",word)))
                pred_word_lst.append(_word)
            allwords = "".join(_char_lst)
            sen = "\n*****************\n> in sentences:\n %s \n\n> we marked:\t %s \n\n> and pred:\t %s\n"%(allwords, basewords, predwords)
            with open("/home/distdev/bilstm/hund.txt", "a+") as f:
                f.write(sen)
                _print(sen)
            return allwords, basewords, predwords

    def run(self):
        result, _, _= self.predict(sent) 
        for sent in result:
            _char_lst, _tags_pred_lst, _y_batch_lst = sent[0], sent[1], sent[2]
            _print(_char_lst, _tags_pred_lst, _y_batch_lst)
            assert len(_char_lst) == len(_tags_pred_lst)
            assert len(_char_lst) == len(_y_batch_lst)
            item_pred, item_base = "", ""
            for i,j,k in zip(_char_lst, _tags_pred_lst, _y_batch_lst):
                item_pred += "%s/%s "%(i, j)
                item_base += "%s/%s "%(i, k)
            basewords = list(re.findall("(./[arvd] (?:.*?)) ./[bies] ", item_base))
            base_word_lst = []
            for word in basewords:
                _word = "".join(list(re.findall("(.)/. ",word)))
                base_word_lst.append(_word)
            predwords = list(re.findall("(./[ardv] (?:.*?)) ./[bies] ", item_pred))
            pred_word_lst = []
            for word in predwords:
                _word = "".join(list(re.findall("(.)/.  ",word)))
                pred_word_lst.append(_word)

            allwords = "".join(_char_lst)
            sen = "\n*****************\n> in sentences:\n %s \n\n> we marked:\t %s \n\n> and pred:\t %s\n"%(allwords, basewords, predwords)
            with open("/home/distdev/bilstm/gz_gongan_case_predict_crim_addr_ext.txt", "a+") as f:
                f.write(sen)
                _print(sen)
            return allwords, basewords, predwords

    def words_pick(self, basewords, predwords, allwords):
        _words = []
        _words.extend(["".join(re.findall('(.)/.',i)) for i in predwords])
        words_set = _words
        if words_set ==[]:
            return []
        lgr.info(words_set)
        lgr.info(str(words_set))
        bg, ed = [], []
        _final_addr = []
        reg = ""
        for i in words_set:
            reg+="%s"%i
            reg+="(.*?)"
        reg = reg[:-5]
        result = []
        if len(words_set) ==1:
            result = words_set
            print(result)
        elif len(words_set) ==0:
            result = []
            print(result)
        elif len(words_set) ==2:
            result = list(re.findall(reg, allwords))
            print(result)
        else:
            result = list(re.findall(reg, allwords)[0])
            assert len(result) == len(words_set)-1
            print(result)
        print(result)
        result_words = []
        _wd = ""
        for i in range(len(words_set)):
            _wd+=words_set[i]
            if i == len(words_set)-1:
                result_words.append(_wd)
                print(_wd)
                break
            else:
                if len(result[i])<5:
                    _wd+=result[i]
                else:
                    result_words.append(_wd)
                    _wd = ""
                    print(_wd)
                    _wd = ""
        return result_words

    def pos_word(self, sent):
        posword = jieba.posseg.cut(sent)
        words_ = ""
        for word in posword:
            words_+="%s/%s "%(word.word, word.flag)
        return words_

    def pos_word_addr(self,sent):
        words_ = self.pos_word(sent)
        lst_  = re.findall("((?:(.)/(?:ns|nr|nz) ){1,})", words_)
        words_lst = []
        for i in lst_:
            words_lst.append(''.join(i))
        return words_lst

    def readDoc(self, fname):
        f = open(_path(fname))
        lines = f.read().split("\n")
        return lines

if __name__ == "__main__":
    sents = []
    eval_ins = Eval_Ner()
    #coll = pymongo.MongoClient("mongodb://127.0.0.1:27017")['myDB']['ner_addr_crim_sample']
    result = eval_ins.predict_txt(eval_ins.readDoc("gz_jyaq.txt"))
    print(result)


