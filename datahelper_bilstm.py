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
        self.train_data_generator = self.gen_train_data(per=3,name='train')
        self.eval_data_generator = self.gen_train_data(per=3, name="eval")
        self.tags = {'b':1,'i':2,'e':3,'s':4,'a':5,'d':6,'r':7,'v':0}# words bg mid end / addrs bg mid end

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

    def arctic_get_all(self):
        # handle the pd data, like panel dataframe series
        libs_name = self.store.list_libraries()
        for lib_name in libs_name:
            self.arc_libs[lib_name] = self.store[lib_name]
        for _lib in self.arc_libs.keys():
            colls_name = self.arc_libs[_lib].list_symbols()
            for col_name in colls_name:
                if col_name in self.arc_colls.keys():
                    _print("\n there r two colls has the same name, the old one is be cover, modify the coll name or use lib.coll to request")
                _print(_lib)
                self.arc_colls[col_name] = store[_lib].read(col_name).data
        _print("\n there r %s libs and %s colls"%(len(self.lib), len(self.arc_colls)))
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

    def marker_target(self, txt):#avd r
        words_marker = ""
        words = list(jieba.cut(txt))
        for word in words:
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
        input data is a (1000,8) array 
        """
        assert data.shape == (1000,8)
        output = []
        for i in range(0,1000):
            for j in range(i-m,i+m):
                if j<0 or j>999:
                    output.extend([0.0]*8)
                else:
                    output.extend(data[j,:])
        #pdb.set_trace()
        #print(np.array(output).reshape(1000,8*(n-1))) 
        return np.array(output).reshape(1000,8*(n-1))    


    def gen_train_data(self, per=3, name="train"):
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
                _print(_crim, _text)
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
                _print(a,b)
            except StopIteration:
                traceback.print_exc()
                round_cnt=0
                self.train_data_generator = self.gen_train_data(per=0.8,name=flag)
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

    def gen_test_data(self):
        """
        generate the test data to feed train model, use memory as small as posible
        """
        pass
