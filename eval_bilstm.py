import sys
import gensim
import pdb
import pymongo
import traceback
from dmp.gongan.ssc_dl_ner.data_utils import full_to_half
import numpy as np
import pandas as pd
import logging
import pdb
import re
import pdb
from tqdm import tqdm
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
print(CURPATH)
sys.path.append(PARPATH)
sys.path.append(CURPATH)
#from bilstm import addr_classify
#from bilstm import eval_bilstm
import bilstm
from bilstm import datahelper
DEBUG =True
DATA = True
import gensim

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

def _path(filepath):
    CURPATH = os.path.dirname(os.path.realpath(__file__))
    _print("\n> CURPATH IS ", CURPATH)
    return os.path.join(CURPATH, filepath)
class Eval_Ner(object):
    def __init__(self):
        """
        # this class is for test Bilstm_Ner model
        """
        self.data_helper=bilstm.datahelper.Data_Helper()
        #ckpt = tf.train.get_checkpoint_state('./model/')
        #saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path +'.meta')
        #print(saver)
        #self.meta_graph_path = saver #_path("model/bilstm.ckpt-7.meta")
        #self.model_path = ckpt.model_checkpoint_path #_path("model/bilstm.ckpt-7")
        #print(self.model_path)
        self.init_eval_graph()
        self.tags = {'o':0,'b':1,'i':2}# words bg mid end / addrs bg mid end
        self.rev_tags = dict(zip(self.tags.values(), self.tags.keys()))

    def tag_map(self, pred_lst_1d):#[0:7]
        #_print(pred_lst_1d)
        _ = list(pred_lst_1d)
        return self.rev_tags[_.index(max(_))]

    def init_eval_graph(self):
        tfconfig = tf.ConfigProto()
        tfconfig.gpu_options.allow_growth = True
        #tfconfig.device_count= {'cpu':0}
        self.sess = tf.Session(config=tfconfig)
        self.sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state('./model/')
        saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path +'.meta')
        saver.restore(self.sess, ckpt.model_checkpoint_path)
        graph = tf.get_default_graph()
        self.X_inputs=tf.get_collection("model.X_inputs")[0]
        self.y_inputs=tf.get_collection("model.y_inputs")[0]
        #self.y_pred_meta=tf.get_collection("model.y_pred")[0]
        self.y_pred_meta=tf.get_collection("attention")[0]
        self.lr=tf.get_collection("lr")[0]
        self.batch_size=tf.get_collection("batch_size")[0]
        self.keep_prob=tf.get_collection("keep_prob")[0]
        self.correct_prediction = tf.equal(tf.cast(tf.argmax(self.y_pred_meta, 1), tf.int32), tf.reshape(self.y_inputs, [-1]))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        #pdb.set_trace()

    def predict_sent(self, sent):
        n=1
        _acc, _acc_average =  0.0, 0.0
        _y_batch_lst = []
        _lr = 1e-4
        start_time = time.time()
        sent_data = []
        for i in range(2000):
            if i <len(sent):
                try:
                    wid = self.data_helper.dct.doc2idx([sent[i]])
                    if wid[0]<0:
                        wid = [215]
                    sent_data.extend(wid)
                except:
                    wid = [215]
                    sent_data.extend(wid)
            else:
                sent_data.extend([215])
        #pdb.set_trace()
        X_batch, y_batch = np.array(sent_data).reshape(32,200), np.array([0]*2000).reshape(32,200)
        _print(X_batch.shape)
        _print(y_batch.shape)
        feed_dict = {self.X_inputs:X_batch, self.y_inputs:y_batch, self.lr:1e-4, self.batch_size:10, self.keep_prob:1.0}
        #_print("y_pred 预测值是:", sess.run(y_pred_meta, feed_dict=feed_dict))
        fetches = [self.correct_prediction, self.y_pred_meta, self.accuracy]
        _corr, _y_pred_meta, _acc = self.sess.run(fetches, feed_dict=feed_dict) # the cost is the mean cost of one batch
        #viterbi_out = viterbi(_y_pred_meta)
        _print("\n> _y_pred_meta", _y_pred_meta, _y_pred_meta.shape)
        _print("\n> _y_pred_tags", [self.tag_map(i) for i in _y_pred_meta])
        _print("\n> _acc", _acc)
        _print("\n> corr", _corr)
        self.y_inputs=tf.get_collection("model.y_inputs")[0]
        #_print("\n> _acc_average", (_acc_average+_acc)/cnt)
        result = []
        chars_, yin_, pred_ = "","",""
        cnt = 0
        for i,j,p in zip(list(X_batch.reshape(2000)), list(y_batch.reshape(2000)), _y_pred_meta.reshape(2000,8)):
           chars_+=self.data_helper.dct[i]
           yin_+=self.rev_tags[j]
           pred_+=self.tag_map(p)
           cnt+=1
           if cnt%200==0:
               #pdb.set_trace()
               result.append([chars_, yin_, pred_])
               chars_,  pred_ , yin_= "","",""
        return result, y_batch, _y_pred_meta.reshape(2000,8)

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

    def predict(self):
        #with tf.device('/cpu:0'):    
        rec_dict={}
        rec_dict['cnt']=0
        dct=gensim.corpora.Dictionary.load("./model/my.dct.bak")
        self.dct=dct
        datasrc = self.data_helper
        gen = datasrc.gen_train_data("eval")
        batch_gen=datasrc.next_batch_eval(gen)
        n=1
        _acc, _acc_average =  0.0, 0.0
        _y_batch_lst = []
        _lr = 1e-4
        start_time = time.time()
        #import prettytable
        #from prettytable import PrettyTable
        f = open("predictable.txt", "a+")
        #pdb.set_trace()
        #w2vm = gensim.models.word2vec.Word2Vec.load(os.path.join(CURPATH, "model/w2vm"))
        while(1):
            rec_dict['cnt']+=1
            #pdb.set_trace()
            X_batch, y_batch, W_batch=batch_gen.__next__()
            #pdb.set_trace()
            #X_batch, y_batch = datasrc.next_batch('train')
            #pdb.set_trace()
            _print(X_batch.shape)
            _print(y_batch.shape)
            feed_dict = {self.X_inputs:X_batch, self.y_inputs:y_batch, self.lr:1e-4, self.batch_size:32, self.keep_prob:1.0}
            #_print("y_pred 预测值是:", sess.run(y_pred_meta, feed_dict=feed_dict))
            fetches = [self.correct_prediction, self.y_pred_meta, self.accuracy]
            _corr, _y_pred_meta, _acc = self.sess.run(fetches, feed_dict=feed_dict) # the cost is the mean cost of one batch
            #viterbi_out = viterbi(_y_pred_meta)
            _print("\n> _y_pred_meta", _y_pred_meta, _y_pred_meta.shape)
            #_print("\n> _y_pred_tags", [self.tag_map(i) for i in _y_pred_meta])
            #_print("\n> _acc", _acc)
            _print("\n> corr", _corr)
            #_print("\n> _acc_average", (_acc_average+_acc)/cnt)
            y_=np.argmax(_y_pred_meta.reshape(6400,3),1).reshape(32,200)
            # pdb.set_trace()
            y=y_batch.reshape(32,200)
            x=X_batch.reshape(32,200)
            w=W_batch.reshape(32,200)
            _print("==========================")
            _print("\n> y_ ",y_)
            _print("==========================")
            #judge_lst = []
            for i in range(0,32):
                print("\n> ===================================")
                #table = PrettyTable(["_id","predict/y_inputi/x_input"])
                #table.sort_key("_id")
                #table.reversesort = True
                bs=""
                ss=""
                ts=""
                for m in range(0, 200):
                    #pdb.set_trace()
                    bs+=w[i][m]
                    cnt=200*i+m
                    pred_sample=y_[i][m]# y of pred
                    y_sample=y[i][m]# y of label
                    x_sample=dct.get(x[i][m])#x of id ==> x of word
                    if x_sample==None:
                       x_sample="None"
                       dct.add_documents([["None"]])
                       dct.save('./model/my.dct.bak')
                       #x_sample = "None"
                    #s+="; "
                    assert not x_sample == None
                    if pred_sample==1 or pred_sample==2:
                      ts+=str(x_sample)
                    ss+=x_sample
                    #_print_pred(ss)
                    #_print_pred(ts)
                    #pdb.set_trace() 
                    #judge_para = (y,y_)
                    #table.add_row([cnt,s])
                    #_print_pred("\n> 预测:",y_,",标签:",y,",词语:",x)
                    #judge_lst.append(self.judge(*judge_para))
                    ##print(np.argmax(_y_pred_meta[cnt]), y_batch[i][m], w2vm.similar_by_vector(X_batch[i][m])[0][0])
                    if X_batch[i][m] == 244:
                        break#continue
                f.write("\n> 目标文本: "+ss)
                if ts=="":
                    ts="该行文本未能检出"
                f.write("\n> base text: "+bs+"\n\n")
                f.write("\n> 提取文本: "+ts+"\n\n")
                #print(table)
                #_print(table)
            #if rec_dict['cnt'] %100==0 and rec_dict['cnt']>100:
            #    dct.save("./model/my.dct.bak")
        #pdb.set_trace()
        f.close()
        #return result, y_batch, _y_pred_meta.reshape(2000,8)

    def run_sent(self, sent):
        result, _, _= self.predict_sent(sent)
        for sent in result:
            _char_lst, _tags_pred_lst, _y_batch_lst = sent[0], sent[1], sent[2]
            _print(_char_lst, _tags_pred_lst, _y_batch_lst)
            assert len(_char_lst) == len(_tags_pred_lst)
            assert len(_char_lst) == len(_y_batch_lst)
            #_print(np.array(_tags_pred_lst).shape)
            item_pred, item_base = "", ""
            for i,j,k in zip(_char_lst, _tags_pred_lst, _y_batch_lst):
                item_pred += "%s/%s "%(i, k)
                item_base += "%s/%s "%(i, j)
                item_pred = re.sub(" /v ","",item_pred)
                item_base = re.sub(" /v ","",item_base)
            #pdb.set_trace()
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
            #allwords = list(re.findall("(.)/(?:.) ", item_base))
            #allwords = "".join(allwords)
            sen = "\n*****************\n> in sentences:\n %s \n\n> we marked:\t %s \n\n> and pred:\t %s\n"%(allwords, basewords, predwords)
            #pdb.set_trace()
            with open("/home/distdev/bilstm/hund.txt", "a+") as f:
                f.write(sen)
                _print(sen)
            return allwords, basewords, predwords

    def run(self):
        result, _, _= self.predict()
        for sent in result:
            _char_lst, _tags_pred_lst, _y_batch_lst = sent[0], sent[1], sent[2]
            _print(_char_lst, _tags_pred_lst, _y_batch_lst)
            assert len(_char_lst) == len(_tags_pred_lst)
            assert len(_char_lst) == len(_y_batch_lst)
            #_print(np.array(_tags_pred_lst).shape)
            item_pred, item_base = "", ""
            for i,j,k in zip(_char_lst, _tags_pred_lst, _y_batch_lst):
                item_pred += "%s/%s "%(i, j)
                item_base += "%s/%s "%(i, k)
            #pdb.set_trace()
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
            #allwords = list(re.findall("(.)/(?:.) ", item_base))
            #allwords = "".join(allwords)
            sen = "\n*****************\n> in sentences:\n %s \n\n> we marked:\t %s \n\n> and pred:\t %s\n"%(allwords, basewords, predwords)
            with open("/home/distdev/bilstm/gz_gongan_case_predict_crim_addr_ext.txt", "a+") as f:
                f.write(sen)
                _print(sen)
            return allwords, basewords, predwords

    def words_pick(self, basewords, predwords, allwords):
        print(basewords, predwords, allwords)
        _words = []
        #_words.extend(["".join(re.findall('(.)/.',i)) for i in basewords])
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
        print(reg)
        #pdb.set_trace()
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

if __name__ == "__main__":
    pass
    """
    eval_ins = Eval_Ner()
    for i in range(500):
        allwords, basewords, predwords =  eval_ins.run()
        ws = eval_ins.words_pick(basewords, predwords, allwords)
        print(ws)
        # pdb.set_trace()
        with open("/home/distdev/bilstm/gz_gongan_case_predict_crim_addr_ext.txt", "a+") as f:
            f.write("%s%s"%("final addrs is : ",",".join(ws)))
    """
    sents = []
    #with open("/home/distdev/bilstm/gzbz.txt", 'r') as f:
    #    sents = f.readlines()
    eval_ins = Eval_Ner()
    coll = pymongo.MongoClient("mongodb://127.0.0.1:27017")['myDB']['ner_addr_crim_sample']
    #for i in coll.find():
    #    sent  = i['text']
        #words_lst = eval_ins.pos_word_addr(sent)
        #pdb.set_trace()
        #allwords="受害人：杨芳（女、汉族、1989年12月21日出生、土家族、中专文化程度、身份证号码：522226198912210823、户籍地：贵州省印江县板溪镇杉林村下凯塘组、现住：贵阳市云岩区忠烈街幼师宿舍楼2单元附6号、联系电话：13984064262）到我所报称：2017年11月25日22时许，其行走至省府路贵山苑门口时，发现其上衣包内的一部玫瑰金苹果6PLUS手机被盗，串号：354992075751系王道琴（女，44岁）在西瓜村毛安路154号红雨棚便利店被人盗走一个包，内有约600元现金，报立行政案件。i经民警安鸿兵到现场核实，系报警人严伦（男，汉族，身份证号码：520103193903264016，户籍地：贵州省贵阳市云岩区瑞金西巷34号附11号，现住地：贵阳市云岩区百花山登高路43栋2单元3号，电话：13765849918）2017年11月26日去贵阳市云岩区大营路工商银行取钱时发现其使用的卡号：6222082402001035149工商银行借记卡卡内1000元元人民币与2017年11月23日 "
        #basewords=[]
        #predwords=['4/d 号/r ', '1/a 1/d ', '住/d 地/b ：/e 贵/e 阳/a ', '市/d 云/a ', '岩/d 区/r ', '百/a 花/d ', '山/a 登/a ', '高/d 路/r ', '4/a 3/d ', '栋/r 2/r ', '单/a 元/d ', '3/r 号/b ，/e 电/b 话/i ：/e 1/b 3/i 7/i 6/i 5/i 8/i 4/i 9/i 9/i 1/i 8/e ）/s 2/b 0/i 1/i 7/e 年/s 1/b 1/e 月/s 2/b 6/e 日/s 去/e 贵/b 阳/e 市/s 云/b 岩/e 区/r ', '路/r 工/b 商/e 银/b 行/e 取/b 钱/s 时/s 发/b 现/e 其/s 使/b 用/e 的/s 卡/b 号/e ：/s 6/v ']

        #allwords, basewords, predwords =  eval_ins.run_sent(sent)
        #ws = eval_ins.words_pick(basewords, predwords, allwords)
        #pdb.set_trace()
        #wc = ws.copy()
        #for w in words_lst:
        #    for s in ws:
        #        if not w.find(s) == -1:
        #            wc.append(w)
        #print(allwords, basewords,predwords)
        #with open("/home/distdev/bilstm/hund.txt", "a+") as f:
        #    f.write("> final :" + ",".join(wc))
    eval_ins.predict()


