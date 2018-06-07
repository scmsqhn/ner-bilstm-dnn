import pymongo
import logging
import sys
sys.path.append('/home/distdev')
import bilstm
from bilstm import addr_classify
from bilstm import eval_bilstm
import arctic
import os
import pdb
import gensim
from dmp.gongan.ssc_dl_ner.data_utils import full_to_half
import traceback
#import digital_info_extract as dex
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import re
#from tqdm import tqdm
import time
#import os
import jieba
import collections
#import sklearn.utils
#from sklearn.utils import shuffle
#import myconfig as config
import tensorflow as tf

from addr_classify import Addr_Classify

import sys
import const
Const = const._const()
Const.__setattr__("SUCC", "success")
Const.__setattr__("FAIL", "fail")
Const.__setattr__("FINISH", "finish")
Const.str2var()


STOP_WORD = []
with open("./stop_word.txt","r")as f:
    lines  = f.readlines()
    for line in lines:
        STOP_WORD.append(line)
print(STOP_WORD[:2])

def logging_init():
    logger = logging.getLogger("bilstm_train.logger")
    logger.setLevel(level = logging.INFO)
    handler = logging.FileHandler("/home/distdev/bilstm/logger.log")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
lgr = logging_init()
def _print(*l):
    logger = lgr
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
 '24': 2.000020000200002e-05,\
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
def _path(filepath):
    CURPATH = os.path.dirname(os.path.realpath(__file__))
    _print("\n> CURPATH IS ", CURPATH)
    return os.path.join(CURPATH, filepath)
class Data_Helper(object):

    def __init__(self):
        _print("\ncls Data_Helper instance")
        #assert self.arctic_inf_init() == Const.SUCC
        #self.mongo_inf_init("myDB", "gz_gongan_case")
        self.odd= True
        self.mongo_inf_init("myDB", "gz_gongan_alarm_1617")
        self.dct = gensim.corpora.Dictionary.load("/home/distdev/bilstm/my_dct")
        self.ac =addr_classify.Addr_Classify(["2016年1月1日9时左右，报警人文群华在股市云岩区保利云山国际13栋1楼冬冬小区超市被撬门进入超市盗走现金1200元及一些食品等物品。技术科民警已经出现场勘查。"])
        self.train_data_generator = self.gen_train_data(name='train')
        self.eval_data_generator = self.gen_train_data(name="eval")
        #self.tags = {'x':0.0, 'o':1.0,'a':2.0,'r':3.0,'v':4.0,'d':5.0}
        self.tags = {'o':0,'b':1,'i':2,'e':3,'s':4,'a':5,'d':6,'r':7,'v':8}# words bg mid end / addrs bg mid end

    def common_data_prepare(self):
        """
        prepare the data of common, before train
        """
        pass

    def mongo_inf_init(self, lib_nm, col_nm):
        # handle the mongo db data
        self.conn = pymongo.MongoClient("mongodb://127.0.0.1")
        self.get_mongo_coll(lib_nm, col_nm)
        return Const.SUCC

    def get_mongo_coll(self, lib_nm, col_nm):
        self.mongo_lib = self.conn[lib_nm]
        self.mongo_col = self.mongo_lib[col_nm]
        return self.mongo_col

    def arctic_inf_init(self):
        # handle the pd data, like panel dataframe series
        # with arctic to hadnle mongodb local
        self.store = arctic.Arctic('mongodb://127.0.0.1')
        return Const.SUCC


    def get_arctic_df(self, lib_nm, col_nm):
        collection = self.store[lib_nm].read(col_nm)
        #cnt = collection.data.count()
        dat = collection.data
        return dat

    def marker_sentence(self, txt):#bie s
        words_marker = ""
        words = list(jieba.cut(txt))
        for word in words:
            #if word in STOP_WORD:
            #    continue
            if word == '\r':
                words_marker+=word
            elif word == '\n':
                words_marker+=word
            elif len(word) == 1:
                _ = "%s/s "%word
                words_marker+=_
            elif len(word) == 2:
                _ = "%s/b %s/e "%(word[0], word[1])
                words_marker+=_
            elif len(word)> 2:
                _ = "%s/b "%word[0]
                for i in range(1, len(word)-2):
                    _+="%s/i "%word[i]
                _+="%s/e "%word[-1]
                words_marker+=_
        return words_marker


    def marker_target_pre_aft(self, txt, cont):#avd r
        res = []
        for word in txt:
            l = len(word)
            index = cont.find(word)
            index_lst = [index-2,index-1,index+l+1,index+l+2]
            res.append(index_lst)
        return res

    def marker_target(self, txt):#avd r
        words_marker = ""
        words = list(jieba.cut(txt))
        for word in words:
            #if word in STOP_WORD:
            #    continue
            if word == '\r':
                words_marker+=(word+"/s" )
            elif word == '\n':
                words_marker+=(word+"/s" )
            elif len(word) == 1:
                _ = "%s/r "%word
                words_marker+=_
            elif len(word) == 2:
                _ = "%s/a %s/d "%(word[0], word[1])
                words_marker+=_
            elif len(word)> 2:
                _ = "%s/a "%word[0]
                for i in range(1, len(word)-2):
                    _+="%s/v "%word[i]
                _+="%s/d "%word[-1]
                words_marker+=_
        return words_marker

    def textcnn_data_transform(self, data, n):
        assert n%2 ==1
        m = n//2
        """
        input data is a (1000,9) array
        """
        assert data.shape == (1000,9)
        output = []
        for i in range(0,1000):
            for j in range(i-m,i+m):
                if j<0 or j>999:
                    output.extend([0.0]*8)
                else:
                    output.extend(data[j,:])
        #pdb.set_trace()
        #print(np.array(output).reshape(1000,9*(n-1)))
        return np.array(output).reshape(1000,9*(n-1))

    def gen_train_data_beta(self, name="train"):
        self.get_mongo_coll('myDB','ner_addr_crim_sample')
        #ev = bilstm.eval_bilstm.Eval_Ner()
        _print("\n> gen_train_data new a Eval_Ner()")
        _collections, _cursor,_count = "","",""
        _collections = self.mongo_col
        #get_mongo_coll( 'myDB', "ner_addr_crim_sample")
        _cursor = _collections.find()
        _count = _collections.count()
        begin_cursor,end_cursor = -1,-1
        if name == "train":
            begin_cursor = 100
            end_cursor = _count
        elif name == 'eval':
            begin_cursor = 0
            end_cursor = 100
        _print("begin_cursor, end_cursor")
        _print(begin_cursor)
        _print(end_cursor)
        ll = [i for i in range(begin_cursor, end_cursor)]
        np.random.shuffle(ll)
        for c in ll:
            _crim = _cursor[c]['addrcrim_sum']
            if len(re.findall("[\u4e00-\u9fa5]{2,}",str(_crim)))<1:
                continue
            i = _cursor[c]
            try:
                len(i['addrcrim'])
            except:
                print(i['addrcrim'])
                continue
            if type(_crim) == str:
                _crim = _crim.split(",")
            for _ in _crim.copy():
                if len(_)<3:
                   _crim.remove(_)
            _text = ""
            try:
                _text = _cursor[c]['text']
            except KeyError:
                traceback.print_exc()
            #print(_text)
            _text = full_to_half(_text)
            _text = re.sub("[^\u4e00-\u9fa50-9a-zA-Z]","",_text)
            flag = False
            for i in _text:
                if self.dct.token2id[i] == -1:
                    self.dct.add_documents([[i]])
                    self.dct.save("./my_cnt")
                    _print("\n> char %s is not in dictionary"% i)
                    _text = re.sub(i, " ", _text)
            if len(_text)<3:
                continue
            """
            to test no word be sub 5.11 qinhaining
            txt = re.sub("[^\u4e00-\u9fa5\d\w\n\r\. ]","",_text)
            cri = [re.sub("[^\u4e00-\u9fa5\d\w\r\n\. ]","",i) for i in list(_crim)]
            """
            txt = _text
            cri = []
            cri = [re.sub("[^\u4e00-\u9fa50-9a-zA-Z]","",i) for i in _crim]
            if len(cri)==0:
                continue
            _print(cri)
            #mark_sent = self.marker_sentence(txt)
            words_markers = self.marker_target_pre_aft(cri,  txt)
            a = [i[0] for i in words_markers]
            b = [i[1] for i in words_markers]
            c = [i[2] for i in words_markers]
            d = [i[3] for i in words_markers]
            mark_target_sent = ""
            for i in range(len(txt)):
                if txt[i] == "\r":
                    mark_target_sent += "\r"
                elif txt[i] == "\n":
                    mark_target_sent += "\n"
                elif i in a:
                    mark_target_sent += "%s/a "%txt[i]
                elif i in b:
                    mark_target_sent += "%s/r "%txt[i]
                elif i in c:
                    mark_target_sent += "%s/v "%txt[i]
                elif i in d:
                    mark_target_sent += "%s/d "%txt[i]
                else:
                    mark_target_sent += "%s/o "%txt[i]
            mark_sent_lst = mark_target_sent.split("\n")
            self.get_mongo_coll('myDB','train_data_bio').insert({"text":mark_target_sent})
            _print("\n> mark_sent_lst: ", mark_sent_lst)
            ids_lst, tags_lst = [], []
            for sent in mark_sent_lst:
                tuple_lst = re.findall("(.)/(.) ", sent)
                _ids = [self.dct.doc2idx([i[0]])[0] for i in tuple_lst]
                _tags = [self.tags[i[1]] for i in tuple_lst]
                ids_lst.extend(_ids)
                tags_lst.extend(_tags)
            _print(len(_ids), len(_tags))
            #_print(_ids, _tags)
            assert len(ids_lst) == len(tags_lst)
            yield ids_lst,tags_lst

    def gen_train_data_arf(self, name="train"):
        self.get_mongo_coll('myDB','ner_addr_crim_sample')
        #ev = bilstm.eval_bilstm.Eval_Ner()
        _print("\n> gen_train_data new a Eval_Ner()")
        _collections, _cursor,_count = "","",""
        _collections = self.mongo_col
        #get_mongo_coll( 'myDB', "ner_addr_crim_sample")
        _cursor = _collections.find()
        _count = _collections.count()
        begin_cursor,end_cursor = -1,-1
        if name == "train":
            begin_cursor = 100
            end_cursor = _count
        elif name == 'eval':
            begin_cursor = 0
            end_cursor = 100
        _print("begin_cursor, end_cursor")
        _print(begin_cursor)
        _print(end_cursor)
        ll = [i for i in range(begin_cursor, end_cursor)]
        np.random.shuffle(ll)
        for c in ll:
            _crim = _cursor[c]['addrcrim_sum']
            if len(re.findall("[\u4e00-\u9fa5]{2,}",str(_crim)))<1:
                continue
            i = _cursor[c]
            try:
                len(i['addrcrim'])
            except:
                print(i['addrcrim'])
                continue
            if type(_crim) == str:
                _crim = _crim.split(",")
            for _ in _crim.copy():
                if len(_)<3:
                   _crim.remove(_)
            _text = ""
            try:
                _text = _cursor[c]['text']
            except KeyError:
                traceback.print_exc()
            #print(_text)
            _text = full_to_half(_text)
            _text = re.sub("[^\u4e00-\u9fa50-9a-zA-Z]","",_text)
            flag = False
            for i in _text:
                if self.dct.token2id[i] == -1:
                    self.dct.add_documents([[i]])
                    self.dct.save("./my_cnt")
                    _print("\n> char %s is not in dictionary"% i)
                    _text = re.sub(i, " ", _text)
            if len(_text)<3:
                continue
            """
            to test no word be sub 5.11 qinhaining
            txt = re.sub("[^\u4e00-\u9fa5\d\w\n\r\. ]","",_text)
            cri = [re.sub("[^\u4e00-\u9fa5\d\w\r\n\. ]","",i) for i in list(_crim)]
            """
            txt = _text
            cri = []
            cri = [re.sub("[^\u4e00-\u9fa50-9a-zA-Z]","",i) for i in _crim]
            if len(cri)==0:
                continue
            _print(cri)
            mark_sent = self.marker_sentence(txt)
            for add_cri in cri:
                if len(add_cri)<2:
                    continue
                lgr.debug('add_cri')
                lgr.debug(add_cri, type(add_cri))
                mark_target = self.marker_target(add_cri)
                mark_target_sent = self.marker_sentence(add_cri)
                #_print(mark_target)
                #_print(mark_target_sent)
                try:
                    mark_sent = re.sub(mark_target_sent, mark_target, mark_sent)
                except:
                    pdb.set_trace()
                mark_sent = re.sub(mark_target_sent, mark_target, mark_sent)
            #_print("\n> mark_sent")
            #_print(mark_sent)
            #pdb.set_trace()
            #_print(re.findall("(.+?)/(.+?) ", mark_sent))
            mark_sent_lst = mark_sent.split("\n")
            ids_lst, tags_lst = [], []
            for sent in mark_sent_lst:
                tuple_lst = re.findall("(.)/(.) ", sent)
                _ids = [self.dct.doc2idx([i[0]])[0] for i in tuple_lst]
                _tags = [self.tags[i[1]] for i in tuple_lst]
                ids_lst.extend(_ids)
                tags_lst.extend(_tags)
            _print(len(_ids), len(_tags))
            #_print(_ids, _tags)
            assert len(ids_lst) == len(tags_lst)
            yield ids_lst,tags_lst

    def gen_train_data(self, name="train"):
        self.get_mongo_coll('myDB','ner_addr_crim_sample')
        #ev = bilstm.eval_bilstm.Eval_Ner()
        _print("\n> gen_train_data new a Eval_Ner()")
        _collections, _cursor,_count = "","",""
        _collections = self.mongo_col
        #get_mongo_coll( 'myDB', "ner_addr_crim_sample")
        #_cursor = _collections.find()
        _count = _collections.count()
        begin_cursor,end_cursor = -1,-1
        if name == "train":
            begin_cursor = 100
            end_cursor = _count
        elif name == 'eval':
            begin_cursor = 0
            end_cursor = 100
        _print("begin_cursor, end_cursor")
        _print(begin_cursor)
        _print(end_cursor)
        ll = [i for i in range(begin_cursor, end_cursor)]
        np.random.shuffle(ll)
        for c in ll:
            _cursor = _collections.find()
            _crim = _cursor[c]['addrcrim']
            if len(re.findall("[\u4e00-\u9fa5]{2,}",str(_crim)))<1:
                continue
            i = _cursor[c]
            try:
                len(i['addrcrim'])
            except:
                print(i['addrcrim'])
                continue
            if type(_crim) == str:
                _crim = _crim.split(",")
            for _ in _crim.copy():
                if len(_)<3:
                   _crim.remove(_)
            _text = ""
            try:
                _text = _cursor[c]['text']
            except KeyError:
                traceback.print_exc()
            #print(_text)
            _text = re.sub("[^\u4e00-\u9fa50-9a-zA-Z]","",_text)
            flag = False
            for i in _text:
                if self.dct.token2id[i] == -1:
                    self.dct.add_documents([[i]])
                    self.dct.save("./my_cnt")
                    _print("\n> char %s is not in dictionary"% i)
                    _text = re.sub(i, " ", _text)
            if len(_text)<3:
                continue
            """
            to test no word be sub 5.11 qinhaining
            txt = re.sub("[^\u4e00-\u9fa5\d\w\n\r\. ]","",_text)
            cri = [re.sub("[^\u4e00-\u9fa5\d\w\r\n\. ]","",i) for i in list(_crim)]
            """
            txt = _text
            cri = []
            cri = [re.sub("[^\u4e00-\u9fa50-9a-zA-Z]","",i) for i in _crim]
            if len(cri)==0:
                continue
            _print(cri)
            mark_sent = self.marker_sentence(txt)
            result = self.split(cri,txt)
            coll = self.get_mongo_coll("myDB","train_data_ner")
            coll.insert({"text":mark_sent})
            coll.insert({"text":mark_sent})
            #_print("\n> mark_sent")
            #_print(mark_sent)
            #pdb.set_trace()
            #_print(re.findall("(.+?)/(.+?) ", mark_sent))
            mark_sent_lst = mark_sent.split("\n")
            ids_lst, tags_lst = [], []
            for sent in mark_sent_lst:
                tuple_lst = re.findall("(.)/(.) ", sent)
                _ids = [self.dct.doc2idx([i[0]])[0] for i in tuple_lst]
                _tags = [self.tags[i[1]] for i in tuple_lst]
                ids_lst.extend(_ids)
                tags_lst.extend(_tags)
            _print(len(_ids), len(_tags))
            #_print(_ids, _tags)
            assert len(ids_lst) == len(tags_lst)
            yield ids_lst,tags_lst

    def gen_train_data_old(self, per=3, name="train"):
        #ev = bilstm.eval_bilstm.Eval_Ner()
        _print("\n> gen_train_data new a Eval_Ner()")
        odd = ~self.odd
        self.odd = odd
        _collections, _cursor,_count = "","",""
        if (odd):
            _collections = self.get_mongo_coll( 'myDB', "gz_gongan_alarm_1617")
            _cursor = _collections.find({},{'反馈内容':1,'_id':0})
            _count = _collections.count()
        else:
            _collections = self.get_mongo_coll( 'myDB', "gz_gongan_case")
            _cursor = _collections.find({},{'jyaq':1, '_id':0})
            _count = _collections.count()
        for i in range(_count):
            if name =="train":
                if i %per ==0:
                    continue
            elif name =="eval":
                if i %per !=0:
                    continue
            _text = list(_cursor[i].items())[0][1]
            print(_text)
            _text = re.sub("[^\d\w\p\S]","",_text)
            _text = full_to_half(_text)
            flag = False
            for i in _text:
                if self.dct.token2id[i] == -1:
                    self.dct.add_documents([[i]])
                    self.dct.save("./my_cnt")
                    _print("\n> char %s is not in dictionary"% i)
                    _text = re.sub(i, " ", _text)

            if len(_text)<3:
                continue
            _crim = -1
            try:
                _crim = self.ac.run(_text)['crim']
                #_crim = ev.run_sent(_text)
                _print("_crim, _text")
                #_print(_crim, _text)
            except:
                _print("\n catch sentences", _text)
                traceback.print_exc()
                continue
            #if len(_crim[i])<1:
            #    continue
            #txt = _text[i]
            """
            to test no word be sub 5.11 qinhaining
            txt = re.sub("[^\u4e00-\u9fa5\d\w\n\r\. ]","",_text)
            cri = [re.sub("[^\u4e00-\u9fa5\d\w\r\n\. ]","",i) for i in list(_crim)]
            """
            txt = _text
            cri = [i for i in list(_crim)]
            _print(cri)
            mark_sent = self.marker_sentence(txt)
            for add_cri in cri:
                if len(add_cri)<2:
                    continue
                lgr.debug('add_cri')
                lgr.debug(add_cri, type(add_cri))
                mark_target = self.marker_target_pre_aft(add_cri, _text)
                mark_target_sent = self.marker_sentence(add_cri)
                #_print(mark_target)
                #_print(mark_target_sent)
                mark_sent = re.sub(mark_target_sent, mark_target, mark_sent)
            #_print("\n> mark_sent")
            #_print(mark_sent)
            #pdb.set_trace()
            #_print(re.findall("(.+?)/(.+?) ", mark_sent))
            mark_sent_lst = mark_sent.split("\n")
            ids_lst, tags_lst = [], []
            for sent in mark_sent_lst:
                tuple_lst = re.findall("(.)/(.) ", sent)
                _ids = [self.dct.doc2idx([i[0]])[0] for i in tuple_lst]
                _tags = [self.tags[i[1]] for i in tuple_lst]
                ids_lst.extend(_ids)
                tags_lst.extend(_tags)
            _print(len(_ids), len(_tags))
            _print(_ids, _tags)
            assert len(ids_lst) == len(tags_lst)
            yield ids_lst,tags_lst

    def next_batch(self, flag="train"):
        _gen = ""
        #_print("\n get the next brach")
        if  flag == "train":
            #_print("\n for train")
            _gen = self.train_data_generator
        elif  flag == "eval":
            #_print("\n for eval")
            _gen = self.eval_data_generator
        _ids = []
        _tags = []
        round_cnt = 0
        while(1):
            #_print("next_batch round_cnt", round_cnt)
            round_cnt+=1
            a,b = -1,-1
            try:
                a,b = _gen.__next__()
                _print("\n> a,b the _gen.next() batch")
                #_print(a,b)
            except StopIteration:
                traceback.print_exc()
                round_cnt=0
                self.train_data_generator = self.gen_train_data(name=flag)
                _gen = self.train_data_generator
                continue
            #_gen = self.gen_train_data(per=0.8, name=flag)
            assert len(a) == len(b)
            la = ((len(a)//100)+1)*100
            left = la-len(a)
            _ids.extend(a)
            _tags.extend(b)
            _ids.extend([215]*left)
            _tags.extend([0]*left)
            #_print(_ids, len(_ids))
            #_print(_tags, len(_tags))
            assert len(_ids) == len(_tags)
            #_print(len(_ids))
            if len(_ids)>2000:
                break
        _ids = _ids[:2000]
        _tags = _tags[:2000]
        #_print(len(_ids))
        #_print(len(_tags))
        try:
            pass
            #_print(np.array(_ids).reshape(10,100).shape)
            #_print(np.array(_tags).reshape(10,100).shape)
        except:
            traceback.print_exc()
        return np.array(_ids).reshape(10,200), np.array(_tags).reshape(10,200)

    #def gen_eval_data(self):
    #    self.gen_train_data(per=0.8, name="eval")
    def merge_coll_mongo(self, lib_nm, col_nm, k1, k2):
        # all items merge into 'addrcrim_sum' k1==>k2
        collection = self.get_mongo_coll(lib_nm, col_nm)
        cursor = self.mongo_col.find()
        for i in  cursor:
                print(i)
                l1 = re.sub("[^\u4e00-\u9fa50-9a-zA-Z,]","",str(i[k1])).split(",")
                l2 = re.sub("[^\u4e00-\u9fa50-9a-zA-Z,]","",str(i[k2])).split(",")
                collection.update_one({"_id":i["_id"]},{"$set":{k1:l1}})
                collection.update_one({"_id":i["_id"]},{"$set":{k2:l2}})
                if len(str(i[k1]))>6:
                    #pdb.set_trace()
                    print(">l1: ",l1)
                if len(str(i[k2]))>6:
                    #pdb.set_trace()
                    print(">l2: ", l2)
                items = []
                items.extend(l1)
                items.extend(l2)
                if len(items)>0:
                    print("> items: ",items)
                """
                for itema in items:
                    if len(itema)<3:
                        continue
                    for itemb in items:
                         if len(itemb)<3:
                            continue
                         for p in range(1,3):
                             for q in range(1,3):
                                 b = itema.find(itemb[:-q])
                                 if b>-1:
                                     if itema in items_copy:
                                         items_copy.remove(itema)
                                     if itemb in items_copy:
                                         items_copy.remove(itemb)
                                     items_copy.append(itema[b:]+itemb)
                                     break
                """
                print("\n> items", items)
                if "" in items:
                    items.remove("")
                if items == None:
                    items = []
                if len(items)>0:
                    pass#pdb.set_trace()
                items_copy = items.copy()
                items_copy2 = items.copy()
                print("\n> items_copy", items_copy)
                print("\n> items_copy2", items_copy2)
                #pdb.set_trace()
                for itema in items_copy2:
                    for itemb in items_copy2:
                        if itema == itemb:
                            continue
                        bina_head = itema.find(itemb)
                        if not bina_head == -1 and (itemb in items_copy):
                            items_copy.remove(itemb)
                items_copy = list(set(items_copy))
                print("\n> items_copy", items_copy)
                if len(items_copy2)>0:# and len(items_copy)==0:
                   pass# pdb.set_trace()
                if len(items_copy)>0:# and len(items_copy)==0:
                   pass# pdb.set_trace()
                collection.update_one({"_id":i["_id"]},{"$set":{k2:items_copy}})

    def replace_kws(self, db, col, key, val1,val2):
        self.get_mongo_coll(db, col).update_many({key:val1},{"$set":{key:val2}})

    def clr_pred(self, db, col, key):
        for i in self.get_mongo_coll(db, col).find():
            ik = self.lst_clr(i[key])
            print(self.get_mongo_coll(db, col).update_one({"_id":i["_id"]},{"$set":{key:ik}}))

    def lst_clr(self, inlst):
        if type(inlst) == str:
            inlst = inlst.split(',')
        kws = ['现金','人民币','公安机关','报案','立案','有限责任公司','手机短信','一条.+?','.+?在.+?时.*?$','.+?服务咨询.*?','.+?一诺财富.+?','.+?有限公司.+?','(?:.+?)(离家出走.+?)','.+?通话时.*?','刑事案件','的一部手机','.+?专用发票.*?','一带.+?','.+?发生冲突','.+?打伤', '卖海洛因', '公司上班期间','上卫生间时','一小区内','笔记本电脑','.*?老板[系是].+?','男子贩毒','的.+?里','被.+?一.+?色.*?手机','民警当场抓获','.{0,4}联系电话','.+?定额.+?','.{0,3}犯罪事实',".{0,3}在网上",'报称其家中','单元防盗门','家[中里]的皮包','.+?面值','.+?税务局定额.*?', '.+?财富.+?','.{0,3}被盗.{0,3}','.{0,3}马路.{0,3}','\d+元','的商品要求依法.+?','^票.+?','.+?余人到我队报称', '[\u4e00-\u9fa5]{0,3}受害人','.+?身份证.+?', '.+?年.+?月.+?日.*?', '.*?短信.*?','一.+?','.*?摩托.*?','最里面','左右','其在','时左右','被人','被男子','有人','.+?[月时].+?时.+?','抓获','离开','贩毒','(?:.+?)(持刀.+?)','^[0-9a-zA-Z]+$']
        outlst = []
        for item in inlst:
            if len(item)<5:
                continue
            for kw in kws:
                item = re.sub(kw,"",item)
            if len(item)>0:
                outlst.append(item)
        return outlst

    def gen_test_data(self):
        """
        generate the test data to feed train model, use memory as small as posible
        """
        pass

    def clr(self, dh):
        dh.clr_pred('myDB', 'ner_addr_crim_sample', 'addrcrim_sum')

def combine_all():
    dh = Data_Helper()
    #dh.merge_coll_mongo('myDB', 'ner_addr_crim_sample', "my_pred","addrcrim_sum")
    #dh.merge_coll_mongo('myDB', 'ner_addr_crim_sample', "my_pred_third","addrcrim_sum")
    #dh.merge_coll_mongo('myDB', 'ner_addr_crim_sample', "my_pred_twice","addrcrim_sum")
    dh.merge_coll_mongo('myDB', 'ner_addr_crim_sample', "addrcrim","addrcrim_sum")
    dh.clr(dh)
def clr_addrcrim_sum():
    dh = Data_Helper()
    dh.clr(dh)

    for i in dh.get_mongo_coll('myDB', 'ner_addr_crim_sample').find()[22500:23000]:
        print(i['addrcrim_sum'])

if __name__ == "__main__":
    clr_addrcrim_sum()
    #combine_all()
    pass

