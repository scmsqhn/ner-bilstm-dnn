#coding=utf-8
import pymongo
import logging
import sys
sys.path.append("/home/distdev/src/iba/dmp/gongan")
from bilstm import addr_classify
#from bilstm import eval_bilstm
import pdb
#import arctic
import os
import pdb
import pdb
#import pdb
import gensim
import traceback
#import digital_info_extract as dex
import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt
import re
#from tqdm import tqdm
#import time
#import os
import jieba
import re 
jieba.load_userdict("./model/all_addr_dict.txt")                     #加载自定义词典  
import jieba.posseg as pseg 
#import collections
#import sklearn.utils
#from sklearn.utils import shuffle
#import myconfig as config
#import tensorflow as tf
#
#from addr_classify import Addr_Classify

import sys
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

"""
STOP_WORD = []
with open("./stop_word.txt","r")as f:
    lines  = f.readlines()
    for line in lines:
        STOP_WORD.append(line)
print(STOP_WORD[:2])
"""

def logging_init(filename="./logger.log"):
    logger = logging.getLogger("bilstm_train.logger")
    logger.setLevel(level = logging.INFO)
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

lgr = logging_init()

pred_lgr = logging_init(filename="./eval.log")

def _print_pred(lgrnm=pred_lgr, *l):
    logger = lgrnm
    if type(l) == str:
        logger.info(l)
    if type(l) == list:
        logger.info(str(l))
    if type(l) == tuple:
        logger.info(str(l))

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
        self.btsize=32
        self.mongo_inf_init("myDB", "gz_gongan_alarm_1617")
        self.w2vm = gensim.models.word2vec.Word2Vec.load("./model/w2vm")
        self.dct = gensim.corpora.Dictionary.load("./model/my.dct.bak")
        self.ac =addr_classify.Addr_Classify(["2016年1月1日9时左右，报警人文群华在股市云岩区保利云山国际13栋1楼冬冬小区超市被撬门进入超市盗走现金1200元及一些食品等物品。技术科民警已经出现场勘查。"])
        self.train_data_generator = self.gen_train_data('train')
        self.eval_data_generator = self.gen_train_data("eval")
        #self.tags = {'x':0.0, 'o':1.0,'a':2.0,'r':3.0,'v':4.0,'d':5.0}
        self.tags = {'o':0,'b':2,'i':1}# words bg mid end / addrs bg mid end
        #self.tags = {'o':0,'b':1,'i':2,'e':3,'s':4,'a':5,'d':6,'r':7,'v':8}# words bg mid end / addrs bg mid end

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
    """
    def arctic_inf_init(self):
        # handle the pd data, like panel dataframe series
        # with arctic to hadnle mongodb local
        self.store = arctic.Arctic('mongodb://127.0.0.1')
        return Const.SUCC
    """
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
                self.arc_colls[col_name] = self.store[_lib].read(col_name).data
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
            #if word in STOP_WORD:
            #    continue
            if word == '\r':
                words_marker+=word
            elif word == '\n':
                words_marker+=word
            elif len(word) == 1:
                _ = "%(.+?)/(.) "%word
                words_marker+=_
            elif len(word) == 2:
                _ = "%(.+?)/(.) %(.+?)/(.) "%(word[0], word[1])
                words_marker+=_
            elif len(word)> 2:
                _ = "%(.+?)/(.) "%word[0]
                for i in range(1, len(word)-2):
                    _+="%(.+?)/(.) "%word[i]
                _+="%(.+?)/(.) "%word[-1]
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
                _ = "%(.+?)/(.) "%word
                words_marker+=_
            elif len(word) == 2:
                _ = "%(.+?)/(.) %(.+?)/(.) "%(word[0], word[1])
                words_marker+=_
            elif len(word)> 2:
                _ = "%(.+?)/(.) "%word[0]
                for i in range(1, len(word)-2):
                    _+="%(.+?)/(.) "%word[i]
                _+="%(.+?)/(.) "%word[-1]
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
        pass#pdb.set_trace()
        #print(np.array(output).reshape(1000,9*(n-1)))
        return np.array(output).reshape(1000,9*(n-1))

    def gen_train_data_beta(self, name="train"):
        self.get_mongo_coll('myDB','traindata')
        #ev = bilstm.eval_bilstm.Eval_Ner()
        _print("\n> gen_train_data new a Eval_Ner()")
        _collections, _cursor,_count = "","",""
        _collections = self.mongo_col
        #get_mongo_coll( 'myDB', "traindata")
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
            _text = self.dwc(_text)
            #flag = False
            if len(_text)<3:
                continue
            txt = _text
            cri = []
            cri = [self.dwc(i) for i in _crim]
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
                    mark_target_sent += "%(.+?)/(.) "%txt[i]
                elif i in b:
                    mark_target_sent += "%(.+?)/(.) "%txt[i]
                elif i in c:
                    mark_target_sent += "%(.+?)/(.) "%txt[i]
                elif i in d:
                    mark_target_sent += "%(.+?)/(.) "%txt[i]
                else:
                    mark_target_sent += "%(.+?)/(.) "%txt[i]
            mark_sent_lst = mark_target_sent.split("\n")
            self.get_mongo_coll('myDB','train_data_bio').insert({"text":mark_target_sent})
            _print("\n> mark_sent_lst: ", mark_sent_lst)
            ids_lst, tags_lst = [], []
            assert len(ids_lst) == len(tags_lst)
            yield ids_lst,tags_lst

    def gen_train_data_arf(self, name="train"):
        self.get_mongo_coll('myDB','traindata')
        #ev = bilstm.eval_bilstm.Eval_Ner()
        _print("\n> gen_train_data new a Eval_Ner()")
        _collections, _cursor,_count = "","",""
        _collections = self.mongo_col
        #get_mongo_coll( 'myDB', "traindata")
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
            _text = self.dwc(_text)
            #flag = False
            if len(_text)<3:
                continue
            txt = _text
            cri = []
            cri = [self.dwc(i) for i in _crim]
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
                    pass#pdb.set_trace()
                    pass
                mark_sent = re.sub(mark_target_sent, mark_target, mark_sent)
            #_print("\n> mark_sent")
            #_print(mark_sent)
            pass#pdb.set_trace()
            #_print(re.findall("(.+?)/(.+?) ", mark_sent))
            #mark_sent_lst = mark_sent.split("\n")
            ids_lst, tags_lst = [], []
            assert len(ids_lst) == len(tags_lst)
            yield ids_lst,tags_lst

    def marker(self, text, flag):
        result = ""
        if len(text)=="":
            return result
        if flag:
            print(text)
            words = self.clr_2_lst(text)
            result += "%s/b "%words[0]
            for word in words[1:]:
                result += "%s/i "%word
        else:
            words = self.clr_2_lst(text)
            for word in words:
                result += "%s/o "%word
        print("\n",text,"\n",result)
        return result

    def split(self, lst, text):
        import pdb
        pass#pdb.set_trace()
        result = ""
        target = []

        regex1 = ""
        for word in lst:
            _ = "(.*?)"+word
            regex1+=_
        regex1+="(.*?)$"

        regex2 = ""
        for word in lst[::-1]:
            _ = "(.*?)"+word
            regex2+=_
        regex2+="(.*?)$"
        print(self.dwc(text))
        print(regex1, regex2)
        try:
            target1 = list(re.findall(regex1, self.dwc(text))[0])
        except:
            target1 = []
            #traceback.print_exc()
        try:
            target2 = list(re.findall(regex2, self.dwc(text))[0])
        except:
            target2 = []
            #traceback.print_exc()
        print(target)
        if len(target1)>0:
            target = target1
            lst=lst
        elif len(target2)>0:
            target = target2
            lst=lst[::-1]
        else:
            return result
        assert len(target) == len(lst)+1
        for i,j in zip(target[:-1],lst):
            result+=self.marker(i, False)
            result+=self.marker(j, True)
        result+=self.marker(target[-1], False)
        print("result", result)
        print("text", text)
        print("lst",lst)
        print("target",target)
        import pdb
        pass#pdb.set_trace()
        return result


    def clr_2_lst(self,sent):
        print(sent)
        res = [self.dwc(i) for i in list(jieba.cut(sent))]
        return res

    def clr_2_str(self,sent):
        pass
        sent = self.dwc(sent)
        res = ""
        for i in list(jieba.cut(sent)):
            #if len(re.findall("[\u4e00-\u9fa5CDM]", str(i)))==0:
            #    continue
            #else:
            res+=i
        return res

    def chdwc(self,sent):
        sent = re.sub("[^\u4e00-\u9fa5]","",sent) # marker
        return sent

    def dwc(self,sent):
        sent = re.sub("[^\u4e00-\u9fa5a-z0-9A-Z@\.]","",sent) # marker
        return sent

    def gen_train_data(self, name="train"):
      g = ""
      if name == "train":
          g = self.gen_train()
      else:
          pass#pdb.set_trace()
          g = self.gen_eval()
      return g

    def format_str(self, *para):
        strout=""
        lenth = len(para)
        for i in range(lenth):
            strout+="{%s},"%i
        return strout[:-1].format(*para)

    def random_lst(self, ll):
        np.random.shuffle(ll)
        return ll

    def throw_exception(self, sent):
        raise Exception(sent)

    def _vali_type(self,dat,tp,name): # (dat) type should be (tp)
        try:
            assert type(dat) == tp
            return Const.SUCC
        except AssertionError:
            sent=self.format_str('\n>In function',name,'\n> the type of',dat,'!=equal',tp)
            self.throw_exception(sent)
            return Const.ERROR

    def _vali_equal(self,left,right,relation,name): # left right is equal small or big
        try:
            if relation=="==":
                assert left==right
            elif relation==">":
                assert left>right
            elif relation=="<":
                assert left<right
            return Const.SUCC
        except AssertionError:
            sent=self.format_str('\n>In function',name,'\n> %s and %s is not %s '%(left,right,relation))
            self.throw_exception(sent)
            return Const.ERROR
         
    def _vali_in(self,child,parent,name): # left right is equal small or big
        try:
            if type(parent)==dict:
                assert (child in parent) ==True
            elif type(parent)==list:
                assert (child in parent) ==True
            elif type(parent)==tuple:
                assert (child in parent) ==True
            elif type(parent)==set:
                assert (child in parent) ==True
            return Const.SUCC
        except AssertionError:
            sent=self.format_str('\n>In function',name,'\n> %s is not in %s '%(child, parent))
            self.throw_exception(sent)
            return Const.ERROR

    def _vali_date_lenth(self,dat,lenth,name): # (dat) type should be (tp)
        try:
            assert type(dat) == list or tuple 
        except AssertionError:
            sent = self.format_str('\n>In function',name,'\n> the type of', dat,' has no function len(), only list and tuple has lenth')
            self.throw_exception(sent)
            return Const.ERROR
        try:
            assert len(dat) == lenth
            return Const.SUCC
        except AssertionError:
            sent = self.format_str('\n>In function',name,'\n> the lenth of', dat,'!=equal',lenth)
            self.throw_exception(sent)
            return Const.ERROR
         
    def toLst(self, s):
        if type(s)==list:
            return s
        elif type(s)==str:
            return s.split(",")
        elif type(s)==tuple:
            return list(s)
         
    def toStr(self, s):
        if type(s)==list:
            return ",".join(s)
        elif type(s)==str:
            return s
        elif type(s)==tuple:
            return ",".join(list(s))

    def fromdct(self,word):
        try:
           res=self.dct.token2id[word]
           return res
        except KeyError:
           print("%sis not in the dct"%word)
           self.dct.add_documents([[word]])
           res=self.dct.token2id[word]
           return res

    def gen_eval(self,funcname="gen_eval",columns_name="casdetail",columns_name_tar="",db="myDB",coll="original_data",begin_cursor=0,end_cursor=300):
            _ids,_tags,_words = [],[],[]
            _print("\n> gen_train_data new a Eval_Ner()")
            self.get_mongo_coll(db,coll)
            _collections = self.mongo_col
            _count = _collections.count()
            _cursor = _collections.find()
            ll = self.random_lst([i for i in range(begin_cursor, end_cursor)])
            print("\n>len ll after random: ",len(ll))
            print("\n>ll:",ll[:10])
            while(1):
                for c in ll:
                    i = _cursor[c]
                    _text=self.dwc(i[columns_name])
                    #_crim=self.dwc(i[columns_name_tar])
                    self._vali_type(i,dict,funcname)
                    #print("\n> this is the ",c,'sentence')
                    desdetail_text = self.clr_2_str(_text)
                    if len(re.findall("[\u4e00-\u9fa5]{2,}",desdetail_text))<1:
                        print("\n> text is no here")
                        continue
                    _ids,_tags,_words=[],[],[]
                    for word in list(jieba.cut(desdetail_text)):
                        _ids.append(self.fromdct(word))
                        _tags.append(0)
                        _words.append(word)
                    if len(_tags)>200:
                        _ids=_ids[:200]
                        _tags=_tags[:200]
                        _words=_words[:200]
                    else:
                        disl=200-len(_tags)
                        _ids.extend([self.dct.token2id[" "]]*disl)
                        _tags.extend([0]*disl)
                        _words.extend([" "]*disl)
                    if Const.DEBUG=="True":
                        _print("gen_eval data _ids _tags")
                        _print("_ids,_tags",_ids,_tags)
                        pass#pdb.set_trace()
                    yield _ids,_tags,_words
                    #_ids,_tags,_words = [],[],[]
                #if len(_tags)%(self.btsize*200)==0 and len(_tags)>2:
                #    self.dct.save("./model/my.dct.bak")

    def gen_train(self, begin_cursor=100,db='myDB',coll='traindata',textcol='text',targetcol='addrcrim',funcname='gen_train'):
            _print("\n> gen_train_data new a Eval_Ner()")
            import  pdb
            pass#pdb.set_trace()
            self.get_mongo_coll(db,coll)
            _collections=self.mongo_col
            count=_collections.count()-begin_cursor
            end_cursor=count-begin_cursor
            self._vali_equal(end_cursor,begin_cursor,">",'gen_train')#断言end_cursor>begin_cursor
            #ev = bilstm.eval_bilstm.Eval_Ner()
            #get_mongo_coll( 'myDB', "traindata")
            #_cursor = _collections.find()
            _cursor = _collections.find()
            ll = self.random_lst([i for i in range(begin_cursor, end_cursor)])
            print("\n>len ll after random: ",len(ll))
            print("\n>ll:",ll[:10])
            for c in ll:
                #print("\n> this is the num",c,'sentence')
                self._vali_type(c,int,funcname)#断言c是int型数据格式
                #===== 过滤掉字段不全的文本
                item=_cursor[c]
                import pdb
                pass#pdb.set_trace()
                try:
                  if self._vali_in(targetcol,item,'gen_train') == Const.ERROR or \
                      self._vali_in(textcol,item,'gen_train') == Const.ERROR:
                      print(Const.KEYLOSS)
                      continue
                except:
                    print(Const.KEYLOSS)
                    continue
                #===== 过滤掉字数太少的文本 和　无中文 的文本
                _crim=self.dwc(item[targetcol])
                _text=self.dwc(item[textcol])
                pass#pdb.set_trace()
                if len(re.findall("[\u4e00-\u9fa5]{2,}",str(_crim)))<1 or len(_text)<3:
                    print("\n> filter text uselessness")
                    print(Const.TEXTUSELESS)
                    continue
                #===== 开始构建 Batch
                _ids, _tags = [], []
                tuple_lst = []
                self._vali_type(_crim,str,funcname)
                _crim_lst=self.toLst(_crim)
                #===== 过滤掉无目标词的文本 将满足要求的地址写入 样本list
                if len(_crim_lst)==0:
                    print(Const.TARGETUSELESS)
                    continue
                _crim_res=[]
                for _ in _crim_lst:
                    if len(_)<3:  
                        print(Const.TARGETUSELESS)
                        continue
                    _crim_res.append(self.clr_2_str(_))
                if len(_crim_res)<1:
                    print(Const.TARGETUSELESS)
                    continue
                #=====　文本transform to str and list
                pass#pdb.set_trace()
                txt = self.clr_2_str(_text)
                lsttxt = self.clr_2_lst(_text)
                result = self.split(_crim_res, _text)
                _print(result)
                self._vali_type(result,str,'gen_train')
                tuple_lst = list(re.findall("(.+?)/(.) ", result))
                _print("\n>tuple_lst: ", tuple_lst)
                _ids=[self.fromdct(j[0]) for j in tuple_lst]
                _tags=[self.tags[j[1]] for j in tuple_lst]
                if len(_tags)>200:
                    _ids=_ids[:200]
                    _tags=_tags[:200]
                else:
                    disl=200-len(_tags)
                    _ids.extend([self.dct.token2id[" "]]*disl)
                    _tags.extend([0]*disl)
                if Const.DEBUG=="True":
                    _print("gen_train data _ids _tags")
                    _print("_ids,_tags",_ids,_tags)
                    pass#pdb.set_trace()
                global SAMPLE_CNT
                SAMPLE_CNT.add(c)
                pass#pdb.set_trace()
                _print("\n> there r ", len(SAMPLE_CNT), 'correct sample total here')
                yield _ids,_tags

    def read_file_2d_lst(self,dirpath,filename):
        f = open(os.path.join(dirpath,filename))
        cont = f.read()
        lines = cont.split("\n")

        y_lst = [line.split("\t")[0] for line in lines]
        x_lst = [line.split("\t")[1] for line in lines]
        clr_lines = [self.dwc(line) for line in x_lst]
        cuts_words = [jieba.cut(line) for line in clr_lines]
        self._vali_equal(len(lines), len(cuts_words), "==")
        return cuts_words, y_lst

    def words_2_ids(self,words):
        self._vali_type(words, list)
        while(1):
            try:
                ids = [self.dct.token2id[word] for word in words]
                self._vali_equal(len(ids),len(words),"==","words_2_ids")
                return ids
            except KeyError:
                self.dct.add_documents([words])
                continue

    def get_lb(self):
        filepath = "/home/distdev/iba/dmp/gongan/shandong_crim_classify/data/lb.txt"
        f = open(filepath)
        lines=f.readlines()
        lbs = [re.sub("[^\u4e00-\u9fa5]","",i) for i in lines]
        c2n={}
        n2c={}
        for i,j in enumerate(lbs):
            c2n[j]=i
            n2c[i]=j
        return c2n,n2c
            

    def gen_train_text_classify_from_text(self, begin_cursor=100,dirpath='/home/distdev/src/iba/dmp/gongan/shandong_crim_classify/data',filename='train.txt.bak',textcol='text',targetcol='addrcrim',funcname='gen_train_text_classify_from_text'):
            res = []
            c2n,n2c=self.get_lb()
            _print('this is the func gen_train_text_classify_from_text')
            words2dlst,y_inputs = self.read_file_2d_lst(dirpath,filename)
            count=len(words2dlst)
            end_cursor = count
            self._vali_equal(count,begin_cursor,">","gen_train_text_classify_from_text")
            self._vali_equal(end_cursor,begin_cursor,">",'gen_train_text_classify_from_text')#断言end_cursor>begin_cursor
            #ev = bilstm.eval_bilstm.Eval_Ner()
            #get_mongo_coll( 'myDB', "traindata")
            #_cursor = _collections.find()
            ll = self.random_lst([i for i in range(begin_cursor, end_cursor)])# shuffle list 
            print("\n>len ll after random: ",len(ll))
            print("\n>ll:",ll[:10])
            for c in ll:
                #print("\n> this is the num",c,'sentence')
                self._vali_type(c,int,funcname)#断言c是int型数据格式
                #===== 过滤掉字段不全的文本
                sent = words2dlst[c]
                _words_id = words_2_ids(sent)
                tag = y_inputs[c]
                _tag_id = c2n[tag]
                yield _words_id,_tag_id,words2dlst[c],tag

    def toArr(self,lst,x,y):
        #import pdb
        pass#pdb.set_trace()
        self._vali_date_lenth(lst,x*y,"toArr()")
        return np.array(lst).reshape(x,y)
 
 
    def next_batch_text_classify_train(self,gen):
        i=0
        _ids,_tags,_words,_lbs=[],[],[],[]
        while(1):
            _id,_tag,_word,_lb=gen.__next__()
            self._vali_equal(len(_id), len(_word))
            self._vali_equal(len(_id), 200)
            self._vali_equal(len(_tag), self.btsize)
            _ids.extend(_id)
            _tags.extend(_tag)
            _words.extend(_word)
            _lbs.extend(_lb)
            i+=1
            print("\n>counter:",i)
            if i==self.btsize:
                yield self.toArr(_ids,self.btsize,200), self.toArr(np.one_hot(_tags),self.btsize,18), self.toArr(_words,self.btsize,200), self.toArr(np.one_hot(_ids),self.btsize,18)
                _ids,_tags,_words,_lbs=[],[],[],[]
                i=0
            #import pdb
            pass#pdb.set_trace()

    def next_batch_eval(self,gen):
        i=0
        _ids,_tags,_words=[],[],[]
        while(1):
            _id,_tag,_word=gen.__next__()
            _ids.extend(_id)
            _tags.extend(_tag)
            _words.extend(_word)
            i+=1
            print("\n>counter:",i)
            if i==self.btsize:
                yield self.toArr(_ids,self.btsize,200), self.toArr(_tags,self.btsize,200), self.toArr(_words,self.btsize,200)
                if Const.DEBUG=="True":
                    print("next_batch_eval")
                    pass#pdb.set_trace()
                _ids,_tags,_words=[],[],[]
                i=0
            #import pdb
            pass#pdb.set_trace()

    def next_batch(self, _gen):
        round_cnt = 0
        _ids,_tags = [],[]
        while(1):
            _print("next_batch round_cnt", round_cnt)
            try:
                import pdb
                pdb.set_trace()
                a,b = _gen.__next__()
                _print("\n> a,b the _gen.next() batch")
                _print(a,b)
                #_gen = self.gen_train_data(per=0.8, name=flag)
                assert len(a) == len(b)
                _ids.append(a)
                _tags.append(b)
                assert len(_ids) == len(_tags)
                round_cnt+=1
                if round_cnt%self.btsize==0:
                    if Const.DEBUG=="True":
                        import pdb
                        print("next_batch")
                        pdb.set_trace()
                    pdb.set_trace()
                    yield np.array(_ids).reshape(self.btsize,200), np.array(_tags).reshape(self.btsize,200)
                    _ids,_tags = [],[]
            except StopIteration:
                pdb.set_trace()
                traceback.print_exc()
                #round_cnt=0
                _gen = self.gen_train_data("train")
                continue

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
                    pass#pdb.set_trace()
                    print(">l1: ",l1)
                if len(str(i[k2]))>6:
                    pass#pdb.set_trace()
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
                    passpdb.set_trace()
                items_copy = items.copy()
                items_copy2 = items.copy()
                print("\n> items_copy", items_copy)
                print("\n> items_copy2", items_copy2)
                pass#pdb.set_trace()
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
                   pass#pdb.set_trace()
                if len(items_copy)>0:# and len(items_copy)==0:
                   pass#pdb.set_trace()
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
        dh.clr_pred('myDB', 'traindata', 'addrcrim_sum')

def combine_all():
    dh = Data_Helper()
    #dh.merge_coll_mongo('myDB', 'traindata', "my_pred","addrcrim_sum")
    #dh.merge_coll_mongo('myDB', 'traindata', "my_pred_third","addrcrim_sum")
    #dh.merge_coll_mongo('myDB', 'traindata', "my_pred_twice","addrcrim_sum")
    dh.merge_coll_mongo('myDB', 'traindata', "addrcrim","addrcrim_sum")
    dh.clr(dh)

def clr_addrcrim_sum():
    dh = Data_Helper()
    dh.clr(dh)

    for i in dh.get_mongo_coll('myDB', 'traindata').find()[22500:23000]:
        print(i['addrcrim_sum'])

if __name__ == "__main__":
    import pdb
    n=310
    dh=Data_Helper()
    train_gen=dh.gen_train_data("train")
    eval_gen=dh.gen_train_data("eval")
    e = dh.next_batch_eval(eval_gen)
    t = dh.next_batch(train_gen)
    while(n>0):
        a,b = t.__next__()
        c,d,w = e.__next__()
        pass#pdb.set_trace()
        n-=1
    #a,b=dh.next_batch(gen)
    #clr_addrcrim_sum()
    #combine_all()


