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
#from tensorflow.contrib import rnn
import numpy as np
import json
from sklearn.model_selection import train_test_split
CURPATH = os.path.dirname(os.path.realpath(__file__))
PARPATH = os.path.dirname(CURPATH)
sys.path.append(PARPATH)
sys.path.append(CURPATH)
import datahelper
DEBUG =True
DATA  =True
DEBUG =False
DATA  =False
import gensim

import const
Const = const._const()
Const.__setattr__("SUCC", "\n> success")
Const.__setattr__("FAIL", "\n> fail")
Const.__setattr__("ERROR", "\n> error")
Const.__setattr__("TEXTUSELESS", "\n无效原文 continue")
Const.__setattr__("TARGETUSELESS", "\n无效目标词 continue")
Const.__setattr__("KEYLOSS", "\n无该key continue")
Const.__setattr__("CLASSIFY_BATCH", "\n输出分类样本batch")
Const.__setattr__("DICT_LOST", "\n该词语在词典中并不存在")
Const.__setattr__("DEBUG", "False")
Const.str2var()

global SAMPLE_CNT
SAMPLE_CNT = set()

def logging_init(filename="./logger.log"):
    logger = logging.getLogger("bilstm_train.logger")
    logger.setLevel(level = logging.INFO)
    handler = logging.FileHandler(filename)

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
        saver = tf.train.import_meta_graph(_path('model/crimAddrBilstm.ckpt-3.meta'))
        saver.restore(self.sess, _path('model/crimAddrBilstm.ckpt-3'))
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

    def prtWordInDic(self, sent):
        words = list(jieba.cut(sent))
        result = []
        for word in words:
            _id = self.data_helper.fromdct(word,flag=False)
            if _id==Const.DICT_LOST:
                continue
            result.extend([word])
        return result


    def words2ids(self, words):
        ids = []
        for word in words:
            _id = self.data_helper.fromdct(word,flag=True)
            if _id== Const.DICT_LOST:
                continue
            ids.extend([_id])
        return ids

    def sent2WordsUnit(self, sent, num=200):
        sent = self.data_helper.dwc(sent)
        print(sent)
        words = list(jieba.cut(sent))
        print(words)
        ids = self.words2ids(words)
        tags = []
        tags.extend([0 for i in range(num)])
        if len(ids) <num:
            lenSub = num - len(ids)
            ids.extend([self.data_helper.fromdct(" ", flag=True) for i in range(lenSub)])
            # pdb.set_trace()
        assert len(ids[:num]) ==num
        assert type(ids[:num]) ==list
        assert len(ids[:num]) ==len(tags[:num])
        return ids[:num], tags[:num]

    def sent2BatchUnit(self,  sents, num=32):
        xbatch,ybatch = [],[]
        for sent in sents:
            wordsTwoHundred, tagsTwohundred = self.sent2WordsUnit(sent)
            xbatch.extend(wordsTwoHundred)
            ybatch.extend(tagsTwohundred)
            if len(xbatch)%(num*200) == 0 and len(xbatch)>1:
                yield  xbatch, ybatch
                xbatch, ybatch = [],[]
        if len(xbatch)>0:
            ld = num*200 - len(xbatch)
            assert ld%200 == 0
            for i in range(ld//200):
                p,q = self.sent2WordsUnit("  ")
                xbatch.extend(p)
                ybatch.extend(q)
            assert len(xbatch) == num*200
            yield xbatch, ybatch

    def clrSet(self, setItem):
        if "" in setItem:
            setItem.remove("")
        return list(setItem)


    def predict_txt(self, text):
        numText = len(text)
        res = []
        gen = self.sent2BatchUnit(text,32)
        while(1):
            #pdb.set_trace()
            X_in, y_in = -1,-1
            try:
                X_in, y_in = gen.__next__()
            except StopIteration:
                print("> data feed over")
                break
            #pdb.set_trace()
            X_batch, y_batch = np.array(X_in).reshape(32,200), np.array(y_in).reshape(32,200)
            #X_batch, y_batch = X_in, y_in
            feed_dict = {self.X_inputs:X_batch, self.y_inputs:y_batch, self.batch_size:32, self.keep_prob:1.0}
            fetches = [self.y_pred_meta]
            #pdb.set_trace()
            [yPredMeta] = self.sess.run(fetches, feed_dict=feed_dict) # the cost is the mean cost of one batch
            y_=np.argmax(yPredMeta.reshape(6400,3),1).reshape(32,200)
            for i,j in zip(X_batch, y_):
                #pdb.set_trace()
                wordsStr = ""
                for m,n in zip(i,j):
                    wordsStr+="%s/%s "%(self.data_helper.dct.get(m), self.rev_tags[n])
                pickTuple=re.split("[\u4e00-\u9fa5a-zA-Z0-9@\.]*/o",wordsStr)
                addrSet = set()
                for p in pickTuple:
                    ret = "".join(re.findall("([\u4e00-\u9fa5a-zA-Z0-9@\.]*)/[ib]",p))
                    addrSet.add(ret)
                res.append(self.clrSet(addrSet))
                #pdb.set_trace()
        numPred = len(res)
        print(numPred)
        if not numText%32 == 0:
           assert numPred == (numText//32+1)*32
        #pdb.set_trace()
        return res

    def readDoc(self, fname):
        f = open(_path(fname))
        lines = f.read().split("\n")
        return lines

    def eval_db(self):
        f = open("source.txt","r")
        g = open("target.txt","a+")
        lines = f.readlines()
        res = []
        np.random.shuffle(lines)
        r =np.random.randint(len(lines)-100)
        for line in lines[r:r+100]:
            res.append(line)
            if len(res) % 32 == 0 :
                result = self.predict_txt(res)
                for m in range(len(result)):
                    g.write("%s\n==>%s\n\n" % (self.data_helper.dwc(res[m]), result[m]))

    def eval_txt(self):
        #coll = pymongo.MongoClient("mongodb://127.0.0.1:27017")['myDB']['ner_addr_crim_sample']
        lines = self.readDoc("gz_jyaq.txt") # ret lines
        result = self.predict_txt(lines)
        return result

    def wr2file(self,result):
        g = open("target.txt","a+")
        lines = self.readDoc("gz_jyaq.txt") # ret lines
        # pdb.set_trace()
        for i in range(len(lines)):
            k = result[i]
            v = "".join(self.prtWordInDic(lines[i]))
            #pdb.set_trace()
            g.write("%s\n==>%s\n\n" % (k, v))

if __name__ == "__main__":
    sents = []
    #coll = pymongo.MongoClient("mongodb://127.0.0.1:27017")['myDB']['ner_addr_crim_sample']
    #result = eval_ins.predict_txt(eval_ins.readDoc("gz_jyaq.txt"))
    #cnt = 0
    #result = eval_ins.predict_txt(['我在马路边抢到一分钱'])
    """
    client = pymongo.MongoClient("mongodb://127.0.0.1:27017")
    dbcoll = client['myDB']['traindata']
    items = dbcoll.find()
    f = open("source.txt","a+")
    for item in items:
        #pdb.set_trace()
        f.write(str(item['text']))
        f.write("\n")
    """
    eval_ins = Eval_Ner()
    result = eval_ins.eval_txt()
    eval_ins.wr2file(result)


