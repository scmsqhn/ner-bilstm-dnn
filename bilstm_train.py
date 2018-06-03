#!coding=utf8
import sys
import os
from tensorflow.python.ops import variable_scope as vs
import pdb
#import gensim
#from dmp.gongan.ssc_dl_ner.data_utils import full_to_half
import traceback
#import digital_info_extract as dex
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import re
#from tqdm import tqdm
import time
#import os
#import jieba
#import collections
#import sklearn.utils
#from sklearn.utils import shuffle
#import myconfig as config
import sklearn as sk
import tensorflow as tf
#rnn = tf.nn.rnn_cell
from tensorflow.contrib import rnn
#contrib import rnn
#import numpy as np
#import json
#import arctic
#from arctic import Arctic
#import pymongo
#import dmp.gongan.gz_case_address.predict as address_predict
#sys.path.append("/home/distdev/addr_classify")
CURPATH = os.path.dirname(os.path.realpath(__file__))
print(CURPATH)
#import sys
PARPATH = os.path.dirname(CURPATH)
sys.path.append(PARPATH)
sys.path.append(CURPATH)
#from bilstm import addr_classify
#from bilstm import eval_bilstm
import bilstm
from bilstm import datahelper
from bilstm.datahelper import Data_Helper
#from eval_bilstm import Eval_Ner
#from bilstm import auto_encode
#from bilstm import text_cnn
#Eval
import logging
from sklearn.metrics import confusion_matrix
from sklearn.metrics import fbeta_score
#from addr_classify.addr_classify import Addr_Classify
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEBUG =True
DATA = True
#import datetime
"""
envs = dict()
with open("envs.json", "r") as f:
    envs = json.loads(f.read())
"""
import sys
import const
Const = const._const()
Const.__setattr__("SUCC", "success")
Const.__setattr__("FAIL", "fail")
Const.__setattr__("FINISH", "finish")
Const.str2var()

def _path(filepath):
    CURPATH = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(CURPATH, filepath)

def logging_init(loggername, filename):
    logger = logging.getLogger(loggername)
    logger.setLevel(level = logging.INFO)
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

bilstmlgr = logging_init("bilstm","./bilstm_logger.log")
datalgr = logging_init("data","./data_logger.log")
trainlgr = logging_init("train","./train_logger.log")
evallgr = logging_init("eval","./eval_logger.log")

lgrdct = {}
lgrdct['bilstmlgr']=bilstmlgr
lgrdct['datalgr']=datalgr
lgrdct['trainlgr']=trainlgr
lgrdct['evallgr']=evallgr

def _print(*l,name="bilstmlgr"):
    logger = lgrdct[name]
    if type(l) == str:
        logger.info(l)
    if type(l) == list:
        logger.info(str(l))
    if type(l) == tuple:
        logger.info(str(l))

_print("\n cur dir file is ", CURPATH)
_print(Const.SUCC)
_print(Const.FAIL)

"""
#------- paraline struct init for 15001 15002 15003 
# init the paral line struct
global NAME
global server
NAME = "server_15001"
#NAME = "server_15002"
#NAME = "server_15003"
cluster=tf.train.ClusterSpec({  
    "worker": [  
        "103.204.229.74:15001",#格式 IP地址：端口号，第1台机器的IP地址 ,在代码中需要用这台机器计算的时候，就要定义：/job:worker/task:0  
        "103.204.229.74:15002",#格式 IP地址：端口号，第2台机器的IP地址 ,在代码中需要用这台机器计算的时候，就要定义：/job:worker/task:1 
    ],  
    "ps": [  
        "103.204.229.74:15003",#格式 IP地址：端口号，第3台机器的IP地址 ,在代码中需要用这台机器计算的时候，就要定义：/job:ps/task:0  
    ]})  
if NAME == "15001":
    global server
    server = tf.train.Server(cluster,job_name='worker',task_index=0)#找到‘worker’名字下的，task0，也就是机器A
elif NAME == "15002":
    global server
    server = tf.train.Server(cluster,job_name='worker',task_index=1)#找到‘worker’名字下的，task0，也就是机器A
elif NAME == "15003":
    global server
    server = tf.train.Server(cluster,job_name='ps',task_index=0)#找到‘worker’名字下的，task0，也就是机器A
    server.join()

paral_saver = tf.train.Saver()  
paral_summary_op = tf.merge_all_summaries()  
paral_init_op = tf.initialize_all_variables()  
paral_sv = tf.train.Supervisor(init_op=init_op, summary_op=summary_op, saver=saver)  
with paral_sv.managed_session(server.target) as sess:  
    while 1:  
        print sess.run([addwb,mutwb,divwb])  

with tf.device(tf.train.replica_device_setter(worker_device='/job:worker/task:0',cluster=cluster)):
    # do sth here
"""
#------- paraline struct init end

class Train_Bilstm_Ner(object):

    def __init__(self):
        _print("\ncls Train_Bilstm_Ner")
        tfconfig = tf.ConfigProto()
        tfconfig.gpu_options.allow_growth = True
        tfconfig.allow_soft_placement=True #FLAGS.allow_soft_placement
        tfconfig.log_device_placement=False #FLAGS.log_device_placement
        self.sess = tf.Session(config=tfconfig)
        self.datahelper = Data_Helper()
        self.model = Bilstm_Att(self.datahelper)
        self._lr = 1e-3
        _print(self.model.y_pred.shape)
        """
        self.att_layer = text_cnn.TextCNN( \
            sequence_length=32, \
            class_num=8, \
            vocab_size=len(self.datahelper.dct), \
            embedding_size=128,\
            num_filters=128,\
            filter_sizes=[2,3,4], \
            l2_reg_lambda=0.0)
        """
        #self.att_layer.dropout_keep_prob = 0.5
        #self.model_combine()
        #assert self.datahelper.arctic_inf_init() == Const.SUCC
        #assert self.datahelper.mongo_inf_init("myDB", "gz_gongan_alarm_1617") == Const.SUCC
        _print("\ncls Train_Bilstm_Ner init finish.")
        #self.save_graph_meta()

    def model_combine(self):
        """
        conbime two or more model into one to build a more large net structure
        """
        _print("\n> combine the tow model show shape")
        #self.att_layer.dropout_keep_prob
        #self.model.y_input_2d  = tf.placeholder([None, 8], dtype=tf.int32)  # 注意类型必须为 tf.int32
        #_ = tf.one_hot(self.model.y_inputs, 8, dtype=tf.int32)
        #_ = tf.cast(self.model.y_inputs, dtype=tf.float32)
    def model_combine(self):
        """
        conbime two or more model into one to build a more large net structure
        """
        _print("\n> combine the tow model show shape")
        #self.att_layer.dropout_keep_prob
        #self.model.y_input_2d  = tf.placeholder([None, 8], dtype=tf.int32)  # 注意类型必须为 tf.int32
        #_ = tf.one_hot(self.model.y_inputs, 8, dtype=tf.int32)
        #_ = tf.cast(self.model.y_inputs, dtype=tf.float32)

        #_ = tf.convert_to_tensor(_, dtype=tf.float32, name=None)

        #self.model.y_input_2d  = tf.reshape(np.array(_), self.models.y_inputs.get_shape())
        #self.model.y_input_2d  = tf.Tensor(self.model.y_input_2d, dtype=tf.float32)
        #self.textcnn_data_transform(self.attention(data, n))
        self.y_pred = self.att_layer.predictions
        _print("\n> self.y_pred.shape:")
        _print(self.y_pred.shape)

    def compare_predict(self, tags, predict, text):
        return pd.DataFrame([tags, predict, text])

    def test_epoch(self):
        """Testing or valid."""
        _print("\n calcu the test_epoch")
        _batch_size = 10
        batch_num = 10
        fetches = [self.model.attention, self.model.accuracy, self.model.cost, self.model.train_op]
        #start_time = time.time()
        _costs = 0.0
        _accs = 0.0
        for i in range(batch_num):
            _print("\n calcu the batch_num in test_epoch")
            X_batch, y_batch = self.datahelper.next_batch("eval")
            feed_dict = {self.model.X_inputs:X_batch, self.model.y_inputs:y_batch, self.model.lr:1e-5, self.model.batch_size:_batch_size, self.model.keep_prob:1.0}
            #pdb.set_trace()
            _att, _acc, _cost, _ = self.sess.run(fetches, feed_dict)
            _print("\n _att")
            _print(_att)
            _print("\n _acc, _cost")
            _print(_acc, _cost)
            _accs += _acc
            _costs += _cost
            _print("\n _accs, _costs")
            _print(_accs, _costs)
        _print("\n batch_num: ", batch_num)
        _print("\n acc 10个字一组，每组acc求和处以组数")
        mean_acc= _accs / batch_num
        mean_cost = _costs / batch_num
        return mean_acc, mean_cost

    """
    def fit(self, to_fetch)
        X_batch, y_batch = self.datahelper.next_batch()
        feed_dict = {self.model.X_inputs:X_batch, self.model.y_inputs:y_batch, self.model.lr:self.model._lr, self.model.batch_size:10, self.model.keep_prob:0.5}

    """
    def att_train(self):

        # Define Training procedure
        cnn = self.att_layer
        self.att_layer.global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        self.att_layer.grads_and_vars = optimizer.compute_gradients(cnn.loss)
        self.att_layer.train_op = optimizer.apply_gradients(self.att_layer.grads_and_vars, global_step=self.att_layer.global_step)
        self.att_layer.optimizer = optimizer

    def train_step_att(self, x_bh, y_bh, sess):

            """
            A single training step
            """
            _print("\n> run train_step_att")

            fetch =  [self.att_layer.global_step, self.att_layer.train_op, self.att_layer.loss, self.att_layer.accuracy, self.att_layer.predictions]
            _feed_dict= {self.att_layer.input_x: x_bh, self.att_layer.input_y: y_bh, self.att_layer.dropout_keep_prob: 0.5}

            _print(_feed_dict)
            result = sess.run(fetch, feed_dict=_feed_dict)
            #_, _step, _loss, _accuracy = sess.run(fetch, feed_dict)
            #time_str = datetime.datetime.now().isoformat()
            #_print("{}: step {}, loss {:g}, acc {:g}".format(time_str, _l, _a))
            return  result

class Bilstm_Att(object):

    def __init__(self, data_helper):
        pass
        self.init_model_para()
        self.init_placeholder()
        self.datahelper = data_helper
        self.gen=self.datahelper.gen_train_data('train')
        self.batch_gen=self.datahelper.next_batch(self.gen)
        self.cost = self.init_model_struct()
        #saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path +'.meta')
        #print(saver)
        #print(self.model_path)
        self.tags = {'o':0,'b':1,'i':2}# words bg mid end / addrs bg mid end
        self.rev_tags = dict(zip(self.tags.values(), self.tags.keys()))
        self.paraLst = []
        self.paraNameLst = []
        self.last_last_cost = 0.0
        self.basecost = 0.0
        self.b= 0.0
        self.w= 0.0
        self._lr_last_last = 1e-2
        self._lr_last = 1e-2
        print("调整lr",self._lr)

        #self.cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = tf.reshape(self.y_inputs, [-1]), logits = self.attention)) #self.attention))
        tvars=tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), self.max_grad_norm)  # 获取损失函数对于每个参数的梯度
        optimizer = tf.train.AdamOptimizer(learning_rate=self._lr)   # 优化器
        self.train_op = optimizer.apply_gradients(list(zip(grads, tvars)), global_step=tf.train.get_or_create_global_step())

        self.save_graph_meta()



    def init_ckpt(self):
        ckpt = tf.train.get_checkpoint_state('./model/')
        self.model_path = ckpt.model_checkpoint_path #_path("model/bilstm.ckpt-7")
        saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path +'.meta')

    def prtAllPara(self):
        for i,j in zip(self.paraLst,self.paraNameLst):
            _print("\n>%s:  %s "%(j,str(self.prtvar(i))))

    def insertParaDict(self,varName, varValue):
        assert type(varName) == str
        try:
            cnt=-1
            for i,j in zip(self.paraNameLst, self.paraLst):
                cnt+=1
                if varName==i:
                    break
            if cnt < len(self.paraLst)-1:
                self.paraLst[cnt]=varValue
                assert self.paraNameLst[cnt]==varName
            else:
                self.paraNameLst.append(varName)
                self.paraLst.append(varValue)
        except AttributeError:
            traceback.print_exc()
            print("there is no this var ", varName)
            pass

    def getParaDict(self,varName):
        for i,j in zip(self.paraLst, self.paraNameLst):
            if varName == i:
                return j
            elif varName == j:
                return i
        return -1
    def tag_map(self, pred_lst_1d):#[0:7]
        #print(pred_lst_1d)
        _ = list(pred_lst_1d)
        return self.rev_tags[_.index(max(_))]

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

    def init_eval_graph(self):
        tfconfig = tf.ConfigProto()
        tfconfig.gpu_options.allow_growth = True
        self.sess = tf.Session(config=tfconfig)
        self.sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state('./model/')
        saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path +'.meta')
        saver.restore(self.sess,self.model_path)
        #graph = tf.get_default_graph()
        self.X_inputs=tf.get_collection("model.X_inputs")[0]
        self.y_inputs=tf.get_collection("model.y_inputs")[0]
        self.y_pred_meta=tf.get_collection("model.y_pred")[0]
        self.lr=tf.get_collection("lr")[0]
        self.batch_size=tf.get_collection("batch_size")[0]
        self.keep_prob=tf.get_collection("keep_prob")[0]
        self.attention=tf.get_collection("attention")[0]
        self.correct_prediction_bilstm= tf.equal(tf.cast(tf.argmax(self.attention, 1), tf.int32), tf.reshape(self.y_inputs, [-1]))
        self.correct_prediction_attention = tf.equal(tf.cast(tf.argmax(self.y_pred_meta, 1), tf.int32), tf.reshape(self.y_inputs, [-1]))
        self.accuracy_attention = tf.reduce_mean(tf.cast(self.correct_prediction_attention, tf.float32))
        self.accuracy_bilstm = tf.reduce_mean(tf.cast(self.correct_prediction_bilstm, tf.float32))

    def predict(self):
        self.init_eval_graph()
        #n=1
        #_acc = 0.0#_acc, _acc_average =  0.0, 0.0
        #_y_batch_lst = []
        #_lr = 1e-3
        #start_time = time.time()
        X_batch, y_batch = self.batch_size.__next__()
        print(X_batch.shape)
        print(y_batch.shape)
        feed_dict = {self.X_inputs:X_batch, self.y_inputs:y_batch, self.lr:1e-3, self.batch_size:10, self.keep_prob:1.0}
        #print("y_pred 预测值是:", sess.run(y_pred_meta, feed_dict=feed_dict))
        fetches = [self.y_pred_meta,self.attention,self.accuracy_bilstm, self.accuracy_attention]
        _y_pred_meta, _pred_att, _acc_bilstm, _acc_att = self.sess.run(fetches, feed_dict=feed_dict) # the cost is the mean cost of one batch
        #viterbi_out = viterbi(_y_pred_meta)
        _print("\n> _y_pred_meta", _y_pred_meta, _y_pred_meta.shape)
        _print("\n> _y_pred_tags", [self.tag_map(i) for i in _y_pred_meta])
        _print("\n> _acc_bilstm", _acc_bilstm)
        _print("\n> divider====================")
        _print("\n> _pred_att", _pred_att, _pred_att.shape)
        _print("\n> _pred_att_tags", [self.tag_map(i) for i in _pred_att])
        _print("\n> _acc_att", _acc_att)
        #print("\n> _acc_average", (_acc_average+_acc)/cnt)
        result = []
        chars_, yin_, pred_, pred_att_ = "","","",""
        cnt = 0
        for i,j,p,pa in zip(list(X_batch.reshape(2000)), list(y_batch.reshape(2000)), _y_pred_meta.reshape(2000,8), _pred_att.reshape(2000,8)):
           try:
               assert self.datahelper.dct[i] != -1             
           except:
               self.datahelper.dct.add_documents([i])

           chars_+=self.datahelper.dct[i]
           yin_+=self.rev_tags[j]
           pred_+=self.tag_map(p)
           pred_att_+=self.tag_map(pa)
           cnt+=1
           if cnt%100==0:
               #pdb.set_trace()
               result.append([chars_, yin_, pred_,pred_att_])
               chars_, yin_, pred_, pred_att_ = "","","",""
        return result

    def run(self):
        result = self.predict()
        for sent in result:
            _char_lst, _y_batch_lst, _tags_pred_lst, _tags_pred_att = sent[0], sent[1], sent[2], sent[3]
            _print(_char_lst, _y_batch_lst, _tags_pred_lst, _tags_pred_att)
            assert len(_char_lst) == len(_tags_pred_lst)
            assert len(_char_lst) == len(_tags_pred_att)
            assert len(_char_lst) == len(_y_batch_lst)
            #print(np.array(_tags_pred_lst).shape)
            item_pred_att, item_pred, item_base = "","",""
            for i,j,k,m in zip(_char_lst, _tags_pred_lst, _y_batch_lst, _tags_pred_att):
                item_pred += "%s/%s "%(i, j)
                item_base += "%s/%s "%(i, k)
                item_pred_att += "%s/%s "%(i, m)
            #pdb.set_trace()
            basewords = list(re.findall("(./a (?:.*)/r) ", item_base))
            predwords_att = list(re.findall("(./a (?:.*)/r) ", item_pred_att))
            predwords= list(re.findall("(./a (?:.*)/r) ", item_pred))
            base_word_lst = []
            for word in basewords:
                _word = "".join(list(re.findall("(.)/. ",word)))
                base_word_lst.append(_word)
            att_pred_word_lst = []
            for word in predwords_att:
                _word = "".join(list(re.findall("(.)/. ",word)))
                att_pred_word_lst.append(_word)
            pred_word_lst = []
            for word in predwords:
                _word = "".join(list(re.findall("(.)/. ",word)))
                pred_word_lst.append(_word)
            allwords = "".join(_char_lst)
            #allwords = list(re.findall("(.)/(?:.) ", item_base))
            #allwords = "".join(allwords)
            sen = "\n*****************\n> in sentences:\n %s \n\n> we marked:\t %s \n\n> and pred:\t %s\n\n> and pred with attention:\t %s\n"%(allwords, basewords, predwords, predwords_att)
            with open("/home/distdev/bilstm/gz_gongan_case_predict_crim_addr_ext.txt", "a+") as f:
                f.write(sen)
                print(sen)


    def init_model_para(self):
        self.tags = {'o':0, 'b':1,'i':2}
        self.decay = 0.85
        max_len = 200# 句子长度
        self.timestep_size =  200
        self.vocab_size = 1000000# 样本中不同字的个数+1(padding 0)，根据处理数据的时候得到
        self.input_size = 64
        self.embedding_size = 128# 字向量长度
        self.class_num = len(self.tags)
        self.hidden_size = 256# 隐含层节点数
        self.layer_num = 4        # bi-lstm 层数
        self.max_grad_norm = 9.0  # 最大梯度（超过此值的梯度将被裁剪）
        self.model_save_path = _path("model/bilstm.ckpt") # 模型保存位置
        self.checkpoint_path = _path("model")  # 模型保存位置
        _print(self.model_save_path)
        _print(self.checkpoint_path)
        self.lastcost = 0.0
        self.tf_batch_num  = 200000
        self.max_epoch = 20
        self.max_max_epoch = 60
        self.btsize = 32

# ====================== model inout=====================


    """
    def init_embedding(self):
        #with tf.variable_scope("embedding", reuse=True) as embedding_scope:
        #with tf.variable_scope("embedding", reuse=None) as embedding_scope:
        #embedding_scope.reuse_variables()
        with tf.device("/cpu:0"):
            self.embedding = tf.get_variable("embedding", [self.vocab_size, self.embedding_size], dtype=tf.int32) 

            # 将输入seq用embedding表示, shape=[batch, steps, hidden_size]
            #inputs = tf.nn.embedding_lookup(embedding, self._input_data)

        #self.embedding = tf.Variable([self.vocab_size, self.embedding_size], dtype=tf.int32)
        #self.embedding=tf.Variable(np.identity([self.vocab_size,self.embedding_size],dtype=np.int32))
        #self.embedding = tf.get_variable("embedding", [self.vocab_size, self.embedding_size], dtype=tf.float32)
        #self.embedding = tf.get_variable("embedding", [self.vocab_size, self.embedding_size], dtype=tf.float32)

    """
    def weight_variable(self, shape):
        """Create a weight variable with appropriate initialization."""
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self,shape):
        """Create a bias variable with appropriate initialization."""
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def lstm_cell(self):
        pass

    def bilstm_cell(self):
        #cell = rnn.BasicLSTMCell(self.hidden_size)
        cell = rnn.LSTMCell(self.hidden_size, reuse=tf.AUTO_REUSE)#tf.get_variable_scope().reuse)
        return rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)

    def init_bi_lstm(self, vocab_size, embedding_size, X_inputs):

            #with tf.variable_scope('my_lstm_cell', reuse=True) as my_lstm_cell:
            """build the bi-LSTMs network. Return the self.y_pred"""
            # self.X_inputs.shape = [batchsize, self.timestep_size]  -  inputs.shape = [batchsize, self.timestep_size, embedding_size]

            #self.embedding = tf.get_variable("embedding", [self.vocab_size, self.embedding_size], dtype=tf.int32) 
            embedding = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0))

            inputs = tf.nn.embedding_lookup(embedding, X_inputs)
            #inputs = self.X_inputs
            int_inputs = tf.cast(inputs, tf.int32)
            float_inputs = tf.cast(inputs, tf.float32)

            #with tf.variable_scope('bidirectional_rnn',reuse=True) as bidirectional_rnn:
            #    tf.get_variable_scope().reuse_variables()
            #with tf.variable_scope('bidirectional_rnn',reuse=Truetf.get_variable_scope().reuse) as bidirectional_rnn:
            #tf.get_variable_scope().reuse = True
            #bidirectional_rnn.reuse_variables()
            outputs_fw = []
            #state_fw.reuse_variables()

            #with tf.variable_scope('fw', reuse=True) as fw:
            #    tf.get_variable_scope().reuse_variables()

            # ** 1.构建前向后向多层 LSTM
            #cell = rnn.BasicLSTMCell(self.hidden_size)#tf.get_variable_scope().reuse)
            #cell = rnn.LSTMCell(self.hidden_size)#tf.get_variable_scope().reuse)
            #rnncell = rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)
    
            cell_fw = rnn.MultiRNNCell([self.bilstm_cell() for _ in range(self.layer_num)], state_is_tuple=True)
            cell_bw = rnn.MultiRNNCell([self.bilstm_cell() for _ in range(self.layer_num)], state_is_tuple=True)

            # ** 2.初始状态
            initial_state_fw = cell_fw.zero_state(self.batch_size, tf.float32)
            initial_state_bw = cell_bw.zero_state(self.batch_size, tf.float32)

            state_fw = initial_state_fw
            for timestep in range(self.timestep_size):
                #if timestep > 0:
                    #tf.get_variable_scope().reuse_variables()
                # pdb.set_trace()
                (output_fw, state_fw) = cell_fw(float_inputs[:, timestep, :], state_fw)
                outputs_fw.append(output_fw)
            outputs_bw = []
            state_bw = initial_state_bw

            # with tf.variable_scope('bw', reuse=True) as bw:
            #tf.get_variable_scope().reuse_variables()
            float_inputs = tf.reverse(float_inputs, [1])

            for timestep in range(self.timestep_size):
                #if timestep > 0:
                #    tf.get_variable_scope().reuse_variables()
                (output_bw, state_bw) = cell_bw(float_inputs[:, timestep, :], state_bw)
                outputs_bw.append(output_bw)
            outputs_bw = tf.reverse(outputs_bw, [0])
            output = tf.concat([outputs_fw, outputs_bw], 2)
            output_trans = tf.transpose(output, perm=[1,0,2])
            self.bilstm_output = tf.reshape(output_trans, [-1, self.hidden_size*2])
            _print("\n bi_lstm output shape ", self.bilstm_output.shape)
            bilstm_y_pred = self.init_full_conn_layer(self. bilstm_output, self.hidden_size*2, self.class_num)
            return bilstm_y_pred
            #return self.output # [-1, self.hidden_size*2]
     
    def init_multi_layer(self,inp,shape0,shape1,dtype=tf.float32):
        softmax_w = self.weight_variable([shape0, shape1])
        softmax_b = self.bias_variable([shape1])
        outp= tf.multiply(inp, softmax_w) + softmax_b
        return tf.cast(outp,dtype)
     
    def init_full_conn_layer(self,inp,shape0,shape1,dtype=tf.float32):
        softmax_w = self.weight_variable([shape0, shape1])
        softmax_b = self.bias_variable([shape1])
        outp= tf.matmul(inp, softmax_w) + softmax_b
        return tf.cast(outp,dtype)


    def fscore(self, labels, logits): # y y_
        y = tf.cast(tf.reshape(labels, [-1]), tf.int32)
        y_ = tf.cast(tf.argmax(logits, 1), tf.int32)
        ylbEqPred = tf.cast(tf.equal(y,y_),tf.int32)
        ylbEqZero = tf.cast(tf.equal(y,0),tf.int32)
        ylbGthZero = tf.cast(tf.greater(y,0),tf.int32)
        yPredEqZero = tf.cast(tf.equal(y_, 0),tf.int32)
        yPredGthZero = tf.cast(tf.greater(y_, 0),tf.int32)
        TP = tf.reduce_sum(tf.cast(tf.where(tf.greater(yPredGthZero,0*yPredGthZero),ylbGthZero, 0*yPredGthZero), tf.float32))
        FP = tf.reduce_sum(tf.cast(tf.where(tf.greater(yPredGthZero,0),ylbEqZero,0*yPredGthZero), tf.float32)) 
        TN = tf.reduce_sum(tf.cast(tf.where(tf.greater(yPredEqZero,0),ylbEqZero, 0*yPredEqZero), tf.float32))
        FN = tf.reduce_sum(tf.cast(tf.where(tf.greater(yPredEqZero,0),ylbGthZero,0*yPredGthZero), tf.float32)) 
        Precision = tf.divide(TP, tf.add(tf.add(TP,FP),0.01))
        Recall= tf.divide(TP, tf.add(tf.add(TP,FN),0.01))
        Fscore = tf.divide(tf.multiply(Precision, Recall), tf.add(Precision, Recall))
        squareFscore = tf.multiply(Fscore, Fscore)
        floss= tf.divide(10000, squareFscore)

        return TP, FP, TN, FN, Precision, Recall, floss
    
    def init_model_struct(self):
        #self.init_embedding()
        #self.bilstm_output = self.init_bi_lstm()
        y_pred = self.init_bi_lstm(self.vocab_size, self.embedding_size, self.X_inputs)
        mycost = self.init_attention(self.X_inputs, y_pred, self.vocab_size, self.btsize, self.timestep_size, self.embedding_size, self.class_num)
        # adding extra statistics to monitor
        return mycost
        # self.y_inputs.shape = [self.batch_size, self.timestep_size]
        #self.correct_prediction = tf.equal(tf.cast(tf.argmax(self.y_pred, 1), tf.int32), tf.reshape(self.y_inputs, [-1]))
        #pdb.set_trace()
        #xx = tf.reshape(self.y_inputs, [-1])
        #xx = tf.cast(xx, dtype=tf.float32)
        #yy = tf.reshape(self.attention, [-1])
        #print(xx,yy,xx.shape,yy.shape)
        #print(xx.get_shape().ndims, yy.get_shape().ndims)

    def reshape(self, inp, divide):
        shape_all = int(inp.shape[0]) * int(inp.shape[1])
        print(shape_all, divide)
        assert shape_all%divide == 0
        return tf.reshape(inp, (shape_all//divide, divide))

    def init_attention(self, X_inputs, y_pred, vocab_size, btsize, timestep_size, embedding_size, class_num):
            #with tf.variable_scope('attentions'):
            #pdb.set_trace(
            #att_inputs = self.X_input
            #att_inputs = tf.reshape(tf.cast(self.X_inputs,tf.float32), (10, 200))
            self.y_pred = y_pred
            embedding = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0))
            att_inputs = tf.cast(tf.reshape(tf.nn.embedding_lookup(embedding, X_inputs), [btsize, timestep_size*embedding_size]), tf.float32)

            #inputs = tf.nn.embedding_lookup(embedding, X_inputs)
            tvm = timestep_size*embedding_size # 200 * 256
            tnm = timestep_size*class_num      # 200 * 3
            divide = 10*embedding_size
            ## ====== 对每个句子进行全连接运算DNN
            #mid = self.init_full_conn_layer(att_inputs,tvm, tvm*2,tf.float32)
            
            # ====== cut sent to 10 words matrix
            att_inputs_div_ten = self.reshape(att_inputs, divide)

            # ====== dnn for every 10 words 
            mid = self.init_full_conn_layer(att_inputs_div_ten, divide, divide,tf.float32)

            # ====== 将维度缩小输出
            dnn_output_divide = self.reshape(mid, tvm)
            dnn_output= self.init_full_conn_layer(dnn_output_divide, tvm, tnm, tf.float32)
            # dnn_output.shape == self.btsize, tnm

            # ====== 将维度缩小输出 对前面的y_pred使用DNN处理
            #y_pred_trans = tf.reshape(y_pred,[btsize, tnm])
            divideClassNum = 10*class_num
            y_pred_trans = self.reshape(y_pred, divideClassNum)
            ## ====== 对每个句子进行全连接运算DNN
            #y_pred_dnn = self.init_full_conn_layer(y_pred_trans,tnm,tnm,tf.float32)
            rnn_output = self.init_full_conn_layer(y_pred_trans,divideClassNum, divideClassNum ,tf.float32)
            # ====== 将rnn dnn 输出结果合并
            sents_cnt = btsize # 32 sent 有多少个句子
            sent_size = timestep_size*class_num # 200 words 3 predict 有多少个 词*词维度 
            r = tf.reshape(rnn_output,[sents_cnt,sent_size])
            d = tf.reshape(dnn_output,[sents_cnt,sent_size])

            rd_tensor = tf.concat((r,d),1)#10,1200
            double_sent_size = 2*sent_size#rd_tensor.shape[0]

            # ====== 将合并后的dnn再输出
            #rnn_part = self.init_multi_layer(r,1,rdl,tf.float32)
            #dnn_part = self.init_multi_layer(d,1,rdl,tf.float32)
            att_sum = self.init_full_conn_layer(rd_tensor,2*sent_size,4*sent_size,tf.float32)
            att_sum_out = self.init_full_conn_layer(att_sum,4*sent_size,sent_size,tf.float32)
            #att_out= tf.add(rnn_part,dnn_part)
            self.attention = tf.reshape(att_sum_out, (btsize*timestep_size, class_num))
            self.TP, self.FP, self.TN, self.FN ,self.Precision, self.Recall,self.Fscore = self.fscore(labels=self.y_inputs, logits=self.attention)

            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = tf.reshape(self.y_inputs, [-1]), logits = self.attention)) #self.attention))
            self.cost = tf.add(self.Fscore, tf.multiply(0.01, self.loss))

            correct_prediction = tf.equal(tf.cast(tf.argmax(self.attention, 1), tf.int32), tf.reshape(self.y_inputs, [-1]))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            return self.cost
            #self.attention = tf.reshape(attention_out, (2000,3))
            #exp_atten= tf.reshape(attention, (10, 200, 9, 1))
            #filter_shape = [5,256,1,1]
            #ati = tf.nn.embedding_lookup(self.embedding, attention)
            #exp_atten = tf.expand_dims(ati, -1)
            #print("\n> exp_atten shape", exp_atten.shape())
            #W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
            #b = tf.Variable(tf.constant(0.1, shape=(10,200,9,1)), name="b")
            #conv = tf.nn.conv2d(exp_atten, W, strides=[1,1,1,1], padding='SAME')
            ## attention2 = tf.nn.relu(attention)
            ## attention3 = tf.nn.dropout(tf.reshape(attention2, (10,600)),0.8)
            #attentio2 = rnn.DropoutWrapper(attention, output_keep_prob=0.5)
            #softmax_w_att_2 = self.weight_variable([600, 600])
            #softmax_b_att_2 = self.bias_variable([10, 600])
            #attention4 = tf.matmul(attention, softmax_w_att_2) + softmax_b_att_2
            #self.attention = tf.reshape(attention4,(10,600))
            #self.attention = tf.argmax(attention5, 1, name="att5")
            #ed = -1 - stry[::-1].find('2')

    def init_placeholder(self):
        self._lr=1e-3
        self.X_inputs = tf.placeholder(tf.int32, [self.btsize, self.timestep_size], name='X_input')
        self.y_inputs = tf.placeholder(tf.int32, [self.btsize, self.timestep_size], name='y_input')
        self.lr = tf.placeholder(tf.float32, [])
        self.keep_prob = tf.placeholder(tf.float32, [])
        self.batch_size = tf.placeholder(tf.int32, [])  # 注意类型必须为 tf.int32

        #self.Precision = sk.metrics.precision_score(tf.reshape(self.y_inputs, [-1]), self.attention)
        #self.Recall = sk.metrics.recall_score(tf.reshape(self.y_inputs,[-1]), self.attention)
        #self.f1_score = sk.metrics.f1_score(tf.reshape(self.y_inputs,[-1]), self.attention)

        """
        s = tf.cast(tf.reshape(self.y_inputs, [-1]), tf.int32)
        f0 = tf.cast(tf.zeros([2000,1]), tf.int32)
        f1 = tf.cast(tf.ones([2000,1]), tf.int32)
        f2 = tf.add(f1,f1)
        ff0 = tf.cast(tf.equal(s,f0), tf.int32) 
        ff1 = tf.cast(tf.equal(s,f1), tf.int32) 
        ff2 = tf.cast(tf.equal(s,f2), tf.int32) 
        #z = tf.cast(tf.reshape(tf.argmax(self.attention, 1), [-1]), tf.int32)
        loss0=tf.multiply(tf.cast(tf.reduce_sum(tf.multiply(tf.cast(correct_prediction,tf.int32),ff0)), tf.float32), 0.1)
        loss1=tf.multiply(tf.cast(tf.reduce_sum(tf.multiply(tf.cast(correct_prediction,tf.int32),ff1)), tf.float32), 5)
        loss2=tf.multiply(tf.cast(tf.reduce_sum(tf.multiply(tf.cast(correct_prediction,tf.int32),ff2)), tf.float32), 2)
        loss3=tf.add(loss0, loss1)
        self.cost=tf.subtract(200.0, tf.add(loss3, loss2))
        print(self.cost)
        """
        #stry = "".join(self.y_inputs)
        #st = stry.find('1')
        #ed = -1 - stry[::-1].find('2')
        #s = tf.cast(tf.reshape(self.y_inputs, [-1]), tf.int32)
        #z = tf.cast(tf.reshape(tf.argmax(self.attention, 1), [-1]), tf.int32)
        #t = tf.multiply(z,s)
        #print("t", t)
        #c = tf.cast(tf.reshape(tf.one_hot(t, 3), (2000,3)), tf.float32)
        #print("c", c)
        #print("self.attention", self.attention)
        #tf.subtract(s,z)
            
        #self.cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = tf.reshape(self.y_inputs, [-1]), logits = c))
        #self.cost_ = tf.reduce_mean(tf.cast(tf.where(tf.greater(y, y_), (y-y_)*20, (y_-y)), tf.float32))
        #self.cost_base = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = tf.reshape(self.y_inputs, [-1]), logits = self.attention)) #self.attention))
        #self.cost_base = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = tf.reshape(self.y_inputs, [-1]), logits = tf.clip_by_value(self.attention, -1, 1, name=None))) #self.attention))
        #self.cost = tf.reduce_mean(tf.divide(1, self.Fscore))
        #self.cost = tf.cast(self.cost_base,tf.float32)
        #self.cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = tf.reshape(self.y_inputs, [-1]), logits = tf.cast(tf.one_hot(tf.argmax(self.attention,1),3), tf.float32)))
        #tmp_y = tf.reshape(tf.cast(tf.one_hot(self.y_inputs,3), tf.int32), [-1])
        #tmp_y_ = tf.reshape(tf.cast(tf.one_hot(tf.argmax(self.attention,1),3), tf.int32),[-1])
        #self.cost = tf.reduce_mean(tf.equal(tmp_y,tmp_y_))

        # pdb.set_trace()

        # 梯度下降计算
        #self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = tf.reshape(self.y_inputs, [-1]), logits = tf.cast(tf.reshape(tf.argmax(self.y_pred,1), [-1]), dtype=tf.float32)))
#attention))
        #self.train_op = tf.train.GradientDescentOptimizer(1e-3).minimize(self.cost)
        #self.train_op = optimizer.apply_gradients(list(zip(grads, tvars)), global_step=tf.contrib.framework.get_or_create_global_step())

    def mod_lr(self,cost):
        sa = 0.1
        if (cost-self.basecost)>0:
            self._lr = self._lr*sa#(1-sa-sa-sa-sa-sa)
        elif (cost-self.basecost)<0:
            self._lr = self._lr*(1+sa/2)
        if self._lr >0.1:
            self._lr =0.1
        elif self._lr<1e-10:
            self._lr = 1e-10
        self.basecost = cost

    def beta_mod_lr(self,cost):
        sa = 0.1
        cost_deta0 = cost-self.last_cost
        cost_deta1 = self.last_cost-self.last_last_cost
        lr_deta0 = self._lr-self._lr_last
        lr_deta1 = self._lr_last-self._lr_last_last

        if lr_deta0==lr_deta1:
            lr_deta0 = sa
        w = (cost_deta0-cost_deta1)-self.b/(lr_deta0-lr_deta1)
        if w == 0:
            w=1
        b = (cost_deta0-cost_deta1)/w
        self.b = self.b+((b-self.b)/2)
        self.w = w
        n = cost_deta0/w
        if n>10:#封顶0.5
            n=10
        elif n<1:
            n=1
        output_lr = n*sa

        if cost - self.last_cost>0:
            newlr = self._lr+sa
        else:
            newlr = self._lr-(n*sa)

        self.last_last_cost = self.last_cost
        self.last_cost = cost
        self._lr_last_last = self._lr_last
        self._lr_last = self._lr
        self._lr = newlr
        if self._lr>sa:
            self._lr =sa
        elif self._lr<1e-10:
            self._lr = 1e-10
        print("self._lr:",self._lr)

    def save_graph_meta(self):
        tf.add_to_collection('model.y_pred', self.y_pred)
        tf.add_to_collection('model.X_inputs',self.X_inputs)
        tf.add_to_collection('model.y_inputs',self.y_inputs)
        tf.add_to_collection('batch_size',self.batch_size)
        tf.add_to_collection('lr',self.lr)
        tf.add_to_collection('keep_prob',self.keep_prob)
        tf.add_to_collection('TP',self.TP)
        tf.add_to_collection('FP',self.FP)
        tf.add_to_collection('TN',self.TN)
        tf.add_to_collection('FN',self.FN)
        tf.add_to_collection('Precision',self.Precision)
        tf.add_to_collection('Recall',self.Recall)
        tf.add_to_collection('Fscore',self.Fscore)

    def prtvar(self, var):
        try:
            return str(var)
            inp = var
            if type(inp) == list:
                return ",".join(inp)
            elif type(inp) == dict:
                return ",".join(list(inp.items()))
            elif type(inp) == set:
                return ",".join(list(inp))
            elif type(inp) == np.ndarray:
                return ",".join(list(inp.reshape([-1])))
            else:
                return var
        except:
           traceback.print_exc()
           pdb.set_trace()
           _print("Sorry, there is no var named %s"% var)
           return var


    def fit_train(self, sess):
        pass
        #self.att_layer = text_cnn.TextCNN( \
        #    sequence_length=32, \
        #    class_num=8, \
        #    vocab_size=len(data_helper.dct), \
        #    embedding_size=128,\
        #    num_filters=128,\
        #    filter_sizes=[2,3,4,5], \
        #    l2_reg_lambda=0.0)
        #self.att_layer.dropout_keep_prob = 0.5
        #self.model_combine()
        #ckpt = tf.train.get_checkpoint_state('./model/')
        #saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path +'.meta')
        saver = tf.train.Saver(max_to_keep=30)
        #model_path = ckpt.model_checkpoint_path #_path("model/bilstm.ckpt-7")
        #saver.restore(sess,model_path)
        #print(self.model_path)
        #print(saver)
        sess.run(tf.global_variables_initializer())
        #test_fetches = [self.attention, self.accuracy, self.cost, self.train_op, self.y_pred]
        train_fetches = [self.attention, self.accuracy, self.cost, self.train_op, self.Precision,self.Recall,self.Fscore, self.TP, self.FP, self.TN, self.FN]
        #train_att_fetches = [self.att_layer.accuracy, self.att_layer.train_op, self.att_layer.predictions]
        #gen = self.datahelper.gen_train_data("train")
        for epoch in range(self.max_max_epoch):
            #self.datahelper.train_data_generator = self.datahelper.gen_train_data(per=0.8,name='train')
            #start_time = time.time()
            _costs, _accs, show_accs, show_costs  = 0.0, 0.0, 0.0, 0.0
            #_costs, _accs = 0.0, 0.0
            for batch in range(self.tf_batch_num):
                _print('EPOCH %d lr=%g' % (epoch+1, self._lr))
                _acc = 0.0
                X_batch, y_batch = self.batch_gen.__next__()
                feed_dict = {self.X_inputs:X_batch, self.y_inputs:y_batch, self.lr:self._lr, self.batch_size:self.btsize, self.keep_prob:0.5}
                res_att, res_acc, res_cost, res_op, res_precision, res_recall, res_fscore, res_tp, res_fp,res_tn,res_fn \
                    = sess.run(train_fetches, feed_dict) # the self.cost is the mean self.cost of one batch
                #att_feed_dict={self.att_layer.input_x:self.textcnn_data_transform(_att,5), self.att_layer.input_y:tf.reshape(tf.one_hot(y_batch,1),(1000,8)), self.att_layer.dropout_keep_prob:0.5 }
                #_att_acc, _att_op, _att_pred, _ = sess.run(train_att_fetches, att_feed_dict) # the self.cost is the mean self.cost of one batch
                #_print(dict(zip(["_att_acc", "_att_op", "_att_pred"],[ _att_acc, _att_op, _att_pred])))
                self.insertParaDict("res_att", res_att)
                self.insertParaDict("res_acc", res_acc)
                self.insertParaDict("res_cost", res_cost)
                self.insertParaDict("res_op", res_op)
                self.insertParaDict("res_precision", res_precision)
                self.insertParaDict("res_recall", res_recall)
                self.insertParaDict("res_fscore", res_fscore)
                self.insertParaDict("res_tp", res_tp)
                self.insertParaDict("res_fp", res_fp)
                self.insertParaDict("res_tn", res_tn)
                self.insertParaDict("res_fn", res_fn)
                res_pred_tri_sum = np.sum(np.argmax(res_att.reshape(6400,3),1))
                res_pred_tri_mean = np.mean(np.argmax(res_att.reshape(6400,3),1))
                self.insertParaDict("res_pred_tri_sum",np.sum(res_pred_tri_sum))
                self.insertParaDict("res_pred_tri_mean",np.sum(res_pred_tri_mean))
                #_accs += _acc
                #_costs += _cost
                show_accs += res_acc
                show_costs += res_cost
                #_print("show_accs, show_costs, _accs, _costs")
                print("acc cost average")
                if batch%3==1:
                   X_batch, y_batch = self.batch_gen.__next__()
                   feed_dict = {self.X_inputs:X_batch, self.y_inputs:y_batch, self.lr:self._lr, self.batch_size:self.btsize, self.keep_prob:0.5}
                   res_mean_acc = show_accs/3
                   res_mean_cost = show_costs/3
                   self.insertParaDict("res_mean_acc", res_mean_acc)
                   self.insertParaDict("res_mean_cost", res_mean_cost)
                   self.prtAllPara()
                   #if mean_cost-self.basecost>0:
                   #    _print("\n> TRIGGER THE NEW _LR SETTING")
                   #    tvars=tf.trainable_variables()  # 获取模型的所有参数
                   #    grads,_=tf.clip_by_global_norm(tf.gradients(self.cost, tvars), self.max_grad_norm)  # 获取损失函数对于每个参数的梯度
                   #    optimizer=tf.train.AdamOptimizer(learning_rate=self._lr)   # 优化器
                   #    self.train_op=optimizer.apply_gradients(list(zip(grads, tvars)), global_step=tf.contrib.framework.get_or_create_global_step())
                   #self.mod_lr(mean_cost)
                   show_accs=0
                   show_costs=0
                if batch%100==1 and batch>100:
                   save_path = saver.save(sess, self.model_save_path, global_step=(epoch+1))
                   _print('the save path is ', save_path)
                   _print('***************save ok ***************')

    def fit_test(self):
        """
        this is for test
        """

    def fit_eval(self, sess):
        """
        this is for eval
        """
        tfconfig = tf.ConfigProto()
        tfconfig.gpu_options.allow_growth = True
        self.sess = tf.Session(config=tfconfig)
        self.sess.run(tf.global_variables_initializer())
        self.new_saver=tf.train.import_meta_graph(self.meta_graph_path)
        self.new_saver.restore(sess,self.model_path)
        #graph = tf.get_default_graph()
        self.X_inputs=tf.get_collection("model.X_inputs")[0]
        self.y_inputs=tf.get_collection("model.y_inputs")[0]
        self.y_pred_meta=tf.get_collection("model.y_pred")[0]
        self.lr=tf.get_collection("lr")[0]
        self.batch_size=tf.get_collection("batch_size")[0]
        self.keep_prob=tf.get_collection("keep_prob")[0]
        self.attention=tf.get_collection("attention")[0]
        self.correct_prediction_bilstm= tf.equal(tf.cast(tf.argmax(self.attention, 1), tf.int32), tf.reshape(self.y_inputs, [-1]))
        self.correct_prediction_attention = tf.equal(tf.cast(tf.argmax(self.y_pred_meta, 1), tf.int32), tf.reshape(self.y_inputs, [-1]))
        self.accuracy_attention = tf.reduce_mean(tf.cast(self.correct_prediction_attention, tf.float32))
        self.accuracy_bilstm = tf.reduce_mean(tf.cast(self.correct_prediction_bilstm, tf.float32))
        saver = tf.train.Saver(max_to_keep=3)
        saver.restore(sess, tf.train.latest_checkpoint(self.model.checkpoint_path))
        X_batch, y_batch = self.batch_gen.__next__()
        test_fetches = [self.attention, self.accuracy_attention, self.accuracy_bilstm, self.y_pred_meta]
        feed_dict = {self.X_inputs:X_batch, self.y_inputs:y_batch, self.lr:self._lr, self.batch_size:10, self.keep_prob:1.0}
        _att_pred, _att_acc, _bilstm_acc , _bilstm_pred = sess.run(test_fetches, feed_dict)
        print(_att_pred,_bilstm_pred, _att_acc, _bilstm_acc)
        return _att_pred,_bilstm_pred, _att_acc, _bilstm_acc

    
if __name__ == "__main__":
    _print("\n train.py")
    train_bilstm_ner_ins =  Train_Bilstm_Ner()
    #train_bilstm_ner_ins.att_train()
    #train_bilstm_ner_ins.test_train_step_att()
    #df = train_bilstm_ner_ins.get_arctic_df("dataframe", "gz_gongan_case_posseg_cut")
    #train_bilstm_ner_ins.data_helper.marker_the_addr_from_context()
    #    gen = train_bilstm_ner_ins.data_helper.gen_train_data()
    #    gen.__next__()
    train_bilstm_ner_ins.model.fit_train(train_bilstm_ner_ins.sess)
    #train_bilstm_ner_ins.model.fit_train(train_bilstm_ner_ins.sess, train_bilstm_ner_ins.datahelper)
    #train_bilstm_ner_ins.()

    #train_bilstm_ner_ins.model.init_eval_graph()
    #for i in range(2):
    #    _print("\n predict 1 sentence")
    #    train_bilstm_ner_ins.model.run()
    """ ["<_RefVariableProcessor(<tf.Variable 'embedding:0' shape=(1000000, 128) dtype=int32_ref>)>", "<_RefVariableProcessor(<tf.Variable 'multi_rnn_cell/cell_0/lstm_cell/kernel:0' shape=(384, 1024) dtype=float32_ref>)>", "<_RefVariableProcessor(<tf.Variable 'multi_rnn_cell/cell_0/lstm_cell/bias:0' shape=(1024,) dtype=float32_ref>)>", "<_RefVariableProcessor(<tf.Variable 'multi_rnn_cell/cell_1/lstm_cell/kernel:0' shape=(512, 1024) dtype=float32_ref>)>", "<_RefVariableProcessor(<tf.Variable 'multi_rnn_cell/cell_1/lstm_cell/bias:0' shape=(1024,) dtype=float32_ref>)>", "<_RefVariableProcessor(<tf.Variable 'multi_rnn_cell/cell_2/lstm_cell/kernel:0' shape=(512, 1024) dtype=float32_ref>)>", "<_RefVariableProcessor(<tf.Variable 'multi_rnn_cell/cell_2/lstm_cell/bias:0' shape=(1024,) dtype=float32_ref>)>", "<_RefVariableProcessor(<tf.Variable 'multi_rnn_cell/cell_3/lstm_cell/kernel:0' shape=(512, 1024) dtype=float32_ref>)>", "<_RefVariableProcessor(<tf.Variable 'multi_rnn_cell/cell_3/lstm_cell/bias:0' shape=(1024,) dtype=float32_ref>)>", "<_RefVariableProcessor(<tf.Variable 'Variable:0' shape=(512, 3) dtype=float32_ref>)>", "<_RefVariableProcessor(<tf.Variable 'Variable_1:0' shape=(3,) dtype=float32_ref>)>", "<_RefVariableProcessor(<tf.Variable 'Variable_2:0' shape=(25600, 51200) dtype=float32_ref>)>", "<_RefVariableProcessor(<tf.Variable 'Variable_3:0' shape=(51200,) dtype=float32_ref>)>", "<_RefVariableProcessor(<tf.Variable 'Variable_4:0' shape=(51200, 600) dtype=float32_ref>)>", "<_RefVariableProcessor(<tf.Variable 'Variable_5:0' shape=(600,) dtype=float32_ref>)>", "<_RefVariableProcessor(<tf.Variable 'Variable_6:0' shape=(600, 600) dtype=float32_ref>)>", "<_RefVariableProcessor(<tf.Variable 'Variable_7:0' shape=(600,) dtype=float32_ref>)>", "<_RefVariableProcessor(<tf.Variable 'Variable_8:0' shape=(1, 600) dtype=float32_ref>)>", "<_RefVariableProcessor(<tf.Variable 'Variable_9:0' shape=(600,) dtype=float32_ref>)>", "<_RefVariableProcessor(<tf.Variable 'Variable_10:0' shape=(1, 600) dtype=float32_ref>)>", "<_RefVariableProcessor(<tf.Variable 'Variable_11:0' shape=(600,) dtype=float32_ref>)>"]"""


