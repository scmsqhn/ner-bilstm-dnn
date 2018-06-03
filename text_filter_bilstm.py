#!coding=utf-8
import sys
from text_cnn_model import TextCNN as cnn
import os
import pdb
#import gensim
#from dmp.gongan.ssc_dl_ner.data_utils import full_to_half
#import traceback
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
import tensorflow as tf
from tensorflow.contrib import rnn
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
Const.__setattr__("DEBUG", "False")
Const.str2var()

def _path(filepath):
    CURPATH = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(CURPATH, filepath)

def logging_init():
    logger = logging.getLogger("./logger.log")
    logger.setLevel(level = logging.INFO)
    handler = logging.FileHandler("./logger.log")
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

_print("\n cur dir file is ", CURPATH)
_print(Const.SUCC)
_print(Const.FAIL)

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
        self._lr = 0.0
        _print(self.model.y_pred.shape)
        """
        self.att_layer = text_cnn.TextCNN( \
            sequence_length=32, \
            num_classes=8, \
            vocab_size=len(self.datahelper.dct), \
            embedding_size=128,\
            num_filters=128,\
            filter_sizes=[2,3,4], \
            l2_reg_lambda=0.0)
        """
        #self.att_layer.dropout_keep_prob = 0.5
        #self.model_combine()
        assert self.datahelper.arctic_inf_init() == Const.SUCC
        assert self.datahelper.mongo_inf_init("myDB", "gz_gongan_alarm_1617") == Const.SUCC
        _print("\ncls Train_Bilstm_Ner init finish.")
        self.tf_batch_num  = 2000
        self.max_epoch = 5
        self.max_max_epoch = 60

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
            pdb.set_trace()
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
        feed_dict = {self.model.X_inputs:X_batch, self.model.y_inputs:y_batch, self.model.lr:self.model._lr, self.model.batch_size:32, self.model.keep_prob:0.5}

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

    def train(self):
        #self.att_train()
        self.sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=3)
        for epoch in range(self.max_max_epoch): # total epoch
            self.model._lr = 1e-4
            if epoch > self.max_epoch:# make the _lr smaller after 5 epoch train
                self.model._lr = self.model._lr * ((self.model.decay) ** (epoch - self.max_epoch))
                _print('EPOCH %d， lr=%g' % (epoch+1, self.model._lr))
            start_time = time.time()
            _costs, _accs, show_accs, show_costs  = 0.0, 0.0, 0.0, 0.0
            for batch in range(self.tf_batch_num):
                fetches = [self.accuracy, self.cost, self.train_op, self.y_pred]
                X_batch, y_batch = self.datahelper.next_batch()
                feed_dict = {self.X_inputs:X_batch, self.y_inputs:y_batch, self.lr:self._lr, self.batch_size:32, self.keep_prob:0.5}
                X_batch, y_batch = self.datahelper.next_batch()
                _acc, self.cost = 0.0, 0.0
                _acc, _cost, _train_op, _model_y_pred = self.sess.run(fetches, feed_dict)
                # the self.cost is the mean self.cost of one batch
                #_y_pred = self.sess.run([self.model.y_pred], feed_dict)
                # the self.cost is the mean self.cost of one batch
                _cost = self.sess.run(self.model.cost, feed_dict)
                # the self.cost is the mean self.cost of one batch
                _train_op = self.sess.run(self.model.train_op, feed_dict)
                # the self.cost is the mean self.cost of one batch
                _model_acc = self.sess.run(self.model.accuracy, feed_dict)
                # the self.cost is the mean self.cost of one batch
                #_print("x, y")
                #bi_pred = np.array(_y_pred[0])
                #bi_pred = (bi_pred+100)*10
                """
                _bi_pred = []
                _bi_pred.append([0]*8)
                _bi_pred.append(bi_pred[0])
                _bi_pred.append(bi_pred[1])
                for i in range(1,999):
                    _bi_pred.append(bi_pred[i-1])
                    _bi_pred.append(bi_pred[i])
                    _bi_pred.append(bi_pred[i+1])
                _bi_pred.append(bi_pred[998])
                _bi_pred.append(bi_pred[999])
                _bi_pred.append([0]*8)
                """
                #_print(len(bi_pred))
                #_x_ = []
                #_ = collections.deque(maxlen=2)
                #_.append([0]*16)
                #for i in bi_pred:
                #    _.append(i)
                #    _x_.extend(list(_))
                #_x_ = np.array(_x_)
                #_print(_x_.shape)
                #pdb.set_trace()
                #_x_ = _x_.reshape(1000,24)*100
                #_x_ = np.array(bi_pred)
                #print(_x_.shape)
                #_ = tf.cast(self.model.y_pred, dtype=tf.int32)
                #pdb.set_trace()
                #print(_.shape)
                #self.att_layer.input_x = _
                #y_batch = np.array(y_batch)
                #n_values = 8
                #_y_batch_att = np.eye(n_values)[y_batch]
                #_y_att_y_input = _y_batch_att.reshape(1000,8)
                #_x_ = _x_.reshape(1000,8)
                #_y_ = _y_att_y_input.astype(int)
                #_x_ = _x_.astype(int)
                #_print(_x_, _x_.shape)
                #_print(_y_, _y_.shape)
                #[_glb_step, _, _att_loss, _att_acc,_att_pred]= \
                #    self.train_step_att(_x_, _y_, self.sess)
                #_print("\n> tf.argmax(_y_, 1)")
                #_ = self.sess.run(tf.argmax(_y_,1))
                #_print(_)
                #_print("\n> _att_loss, _att_accuracy")
                #_print(_att_loss, _att_acc)
                _print("\n> _acc_bilstm")
                _print(_acc)
                #_print("\n> att_prediction")
                #_print(_att_pred)
                _accs += _acc
                _costs += _cost
                show_accs += _acc
                show_costs += _cost
                #_print(display_batch)5
                if batch % 20 ==0:
                    _print(batch , "********************************")
                    _print(X_batch.shape, y_batch.shape)
                    _print("\n %s of %s" % (batch, self.tf_batch_num))
                    _print("\n _cost: ", _cost)
                    _print("\n _train_op: ", _train_op)
                    _print("\n _model_acc: ",_model_acc)
                    _print("\n _acc: ", _acc)
                    _print("\n _cost: ", _cost)
                    _print("\n _train_op: ", _train_op)
                    _print("\n _model_y_pred: ", _model_y_pred)
                    _print("\n _model_y_pred.shape: ", _model_y_pred.shape)
                """
                global_step = tf.Variable(0, name="global_step", trainable=False)
                optimizer = tf.train.AdamOptimizer(1e-3)
                grads_and_vars = optimizer.compute_gradients(cnn.loss)
                train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
                _, step, summaries, loss, accuracy = sess.run( [self.att_layer_cost.train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy], feed_dict)
                """
            display_batch = 100
            if True:#(batch + 1) % (display_batch) == 3:
                save_path = saver.save(self.sess, self.model.model_save_path, global_step=(epoch+1))
                #saver.restore(sess, tf.train.latest_checkpoint(self.model.checkpoint_path))
                _print('the save path is ', save_path)
                _print('and then restore the model ', self.model.checkpoint_path)
                valid_acc, valid_cost = self.test_epoch()  # valid
                _print('\ttraining acc=%g, cost=%g;  valid acc= %g, self.cost=%g ' % (show_accs / display_batch, show_costs / display_batch, valid_acc, valid_cost))
                #_print('\ttraining acc=%g, self.cost=%g;  valid acc= %g, self.cost=%g ' % (show_accs / display_batch, show_costs / display_batch, valid_acc, valid_self.cost))
                show_accs = 0.0
                show_costs = 0.0
                mean_acc = _accs / self.tf_batch_num
                mean_cost = _costs / self.tf_batch_num
                _print('\tacc=%g, self.cost=%g ' % (mean_acc, mean_cost))
                _print('Epoch acc=%g, self.cost=%g, speed=%g s/epoch' % (mean_acc, mean_cost, time.time()-start_time))
                # testing
                _print('**TEST RESULT:')
                test_acc, test_cost = self.test_epoch()
                _print('**Test acc=%g, cost=%g' % (test_acc, test_cost))
                #_print('**Test %d, acc=%g, self.cost=%g' % (data_test.y.shape[0], test_acc, test_self.cost))

    def test_train_step_att(self):
        with tf.Session() as sess:
            x_batch = np.ones([1000,8]).astype(int)
            y_batch = np.ones([1000,8]).astype(int)
            self.train_step_att(x_batch, y_batch, sess)

class Bilstm_Att(object):

    def __init__(self, data_helper):
        pass
        self.init_model_para()
        self.init_model_struct()
        self.datahelper = data_helper
        self.cnn_class_c2n, self.cnn_class_n2c = self.data_helper.get_lb()
        self.gen=self.datahelper.gen_train_text_classify_from_text()
        #saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path +'.meta')
        #print(saver)
        #print(self.model_path)
        self.tags = {'f':0,'p':1}#f filter 过滤掉 p pass 通过
        self.rev_tags = dict(zip(self.tags.values(), self.tags.keys()))

        self.last_last_cost = 0.0
        self.last_cost = 0.0
        self.b= 0.0
        self.w= 0.0
        self._lr_last_last = 1e-2
        self._lr_last = 1e-2
        self._lr = 1e-2
        print("调整lr",self._lr)

    def init_ckpt(self):
        ckpt = tf.train.get_checkpoint_state('./model/')
        self.model_path = ckpt.model_checkpoint_path #_path("model/bilstm.ckpt-7")
        saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path +'.meta')

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
        for i in range(0,3200):
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
        #_lr = 1e-4
        #start_time = time.time()
        X_batch, y_batch = self.datahelper.next_batch('eval')
        print(X_batch.shape)
        print(y_batch.shape)
        feed_dict = {self.X_inputs:X_batch, self.y_inputs:y_batch, self.lr:1e-4, self.batch_size:32, self.keep_prob:1.0}
        #print("y_pred 预测值是:", sess.run(y_pred_meta, feed_dict=feed_dict))
        fetches = [self.y_pred_meta, self.attention, self.accuracy_bilstm, self.accuracy_attention]
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
            item_pred_att, item_pred, item_base = "","", ""
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
        self.tags = {'p':0.0, 'f':1.0}
        self.decay = 0.85
        max_len = 200# 句子长度
        self.timestep_size =  max_len
        self.vocab_size = 100208# 样本中不同字的个数+1(padding 0)，根据处理数据的时候得到
        self.input_size = 64
        self.embedding_size = 128# 字向量长度
        self.class_num = len(self.tags)
        self.cnn_classify_num = 18
        self.hidden_size = 256# 隐含层节点数
        self.layer_num = 4        # bi-lstm 层数
        self.max_grad_norm = 9.0  # 最大梯度（超过此值的梯度将被裁剪）
        self.model_save_path = _path("model/bilstm.ckpt") # 模型保存位置
        self.checkpoint_path = _path("model")  # 模型保存位置
        _print(self.model_save_path)
        _print(self.checkpoint_path)
        self._lr = 1e-4
        self.tf_batch_num  = 2000
        self.max_epoch = 20
        self.max_max_epoch = 60

# ====================== model inout=====================


    def init_embedding(self):
        with tf.variable_scope("embedding", reuse=True) as embedding_scope:
            self.embedding = tf.get_variable("embedding", [self.vocab_size, self.embedding_size], dtype=tf.float32)
            embedding_scope.reuse_variables()

    def weight_variable(self, shape):
        """Create a weight variable with appropriate initialization."""
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self,shape):
        """Create a bias variable with appropriate initialization."""
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def lstm_cell(self):
        cell = rnn.LSTMCell(self.hidden_size, reuse=tf.get_variable_scope().reuse)
        return rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)

    def init_bi_lstm(self):
        """build the bi-LSTMs network. Return the self.y_pred"""
        # self.X_inputs.shape = [batchsize, self.timestep_size]  -  inputs.shape = [batchsize, self.timestep_size, embedding_size]
        #inputs = tf.nn.embedding_lookup(self.embedding, self.X_inputs)
        inputs = self.X_inputs
        int_inputs = tf.cast(inputs, tf.int32)
        float_inputs = tf.cast(inputs, tf.float32)
        # ** 1.构建前向后向多层 LSTM
        cell_fw = rnn.MultiRNNCell([self.lstm_cell() for _ in range(self.layer_num)], state_is_tuple=True)
        cell_bw = rnn.MultiRNNCell([self.lstm_cell() for _ in range(self.layer_num)], state_is_tuple=True)

        # ** 2.初始状态
        initial_state_fw = cell_fw.zero_state(self.batch_size, tf.float32)
        initial_state_bw = cell_bw.zero_state(self.batch_size, tf.float32)

        with tf.variable_scope('bidirectional_rnn',reuse=tf.AUTO_REUSE) as bidirectional_rnn:
            bidirectional_rnn.reuse_variables()
            outputs_fw = list()
            state_fw = initial_state_fw
            with tf.variable_scope('fw', reuse=tf.AUTO_REUSE) as fw_scope:
                #fw_scope.reuse_variables()
                for timestep in range(self.timestep_size):
                    if timestep > 0:
                        tf.get_variable_scope().reuse_variables()
                    (output_fw, state_fw) = cell_fw(float_inputs[:, timestep, :], state_fw)
                    outputs_fw.append(output_fw)
            outputs_bw = list()
            state_bw = initial_state_bw
            with tf.variable_scope('bw', reuse=tf.AUTO_REUSE) as bw_scope:
                #bw_scope.reuse_variables()
                inputs = tf.reverse(inputs, [1])
                for timestep in range(self.timestep_size):
                    if timestep > 0:
                        tf.get_variable_scope().reuse_variables()
                    (output_bw, state_bw) = cell_bw(float_inputs[:, timestep, :], state_bw)
                    outputs_bw.append(output_bw)
            outputs_bw = tf.reverse(outputs_bw, [0])
            output = tf.concat([outputs_fw, outputs_bw], 2)
            output = tf.transpose(output, perm=[1,0,2])
            self.output = tf.reshape(output, [-1, self.hidden_size*2])
            _print("\n bi_lstm output shape ", self.output.shape)
            return self.output # [-1, self.hidden_size*2]

    def input_placeholder(self):
        with tf.variable_scope('Inputs'):
            self.X_inputs = tf.placeholder(tf.float32, [32, self.timestep_size,self.embedding_size], name='X_input')
            self.y_inputs = tf.placeholder(tf.int32, [32, self.cnn_classify_num], name='y_input')

    def init_outputs(self):
        with tf.variable_scope('outputs'):
            softmax_w = self.weight_variable([self.hidden_size * 2, self.class_num])
            softmax_b = self.bias_variable([self.class_num])
            self.y_pred = tf.matmul(self.bilstm_output, softmax_w) + softmax_b
            #pdb.set_trace()
            _print(self.y_pred.shape)


    def init_attention(self):
        with tf.variable_scope('attentions'):
            #pdb.set_trace(
            att_inputs = tf.reshape(tf.cast(self.X_inputs,tf.float32), (32, self.timestep_size*self.embedding_size))
            #att_inputs = tf.reshape(tf.nn.embedding_lookup(self.embedding, self.X_inputs), (32, 200*128))
            # 10 * 200 * 128
            # 10 * 200 * 9
            softmax_w_att = self.weight_variable([200*128, 200*2])
            softmax_b_att = self.bias_variable([32, 200*2])
            #self.attention = tf.matmul(self.y_pred, softmax_w_att) + softmax_b_att
            dnn_output = tf.matmul(att_inputs, softmax_w_att) + softmax_b_att
            softmax_w_att2 = self.weight_variable([400, 400])
            softmax_b_att2 = self.bias_variable([32, 400])
            y_pred_trans = tf.reshape(self.y_pred,(32, 400))
            rnn_output = tf.matmul(y_pred_trans, softmax_w_att2) + softmax_b_att2


            r = tf.reshape(rnn_output,(32,400))
            d = tf.reshape(dnn_output,(32,400))
            rd_tensor = tf.concat((r,d),1)
            print(rd_tensor.shape)
            assert rd_tensor.shape == (32,800)
            rdl = 800#rd_tensor.shape[0]
            print(rdl)
            w0 =  self.weight_variable([rdl,rdl])
            b0 =  self.bias_variable([32,rdl])
            attention = tf.matmul(rd_tensor, w0)+b0
            w1 =  self.weight_variable([rdl,rdl//2])
            b1 =  self.bias_variable([1,rdl//2])
            attention_out = tf.matmul(attention, w1) + b1
            self.attention = tf.argmax(tf.reshape(attention_out, (6400,2)),1)
            #self.attention = tf.reshape(attention_out, (2000,2))
            #exp_atten= tf.reshape(attention, (32, 200, 9, 1))
            #filter_shape = [5,256,1,1]
            #ati = tf.nn.embedding_lookup(self.embedding, attention)
            #exp_atten = tf.expand_dims(ati, -1)
            #print("\n> exp_atten shape", exp_atten.shape())
            #W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
            #b = tf.Variable(tf.constant(0.1, shape=(32,200,9,1)), name="b")
            #conv = tf.nn.conv2d(exp_atten, W, strides=[1,1,1,1], padding='SAME')
            ## attention2 = tf.nn.relu(attention)
            ## attention3 = tf.nn.dropout(tf.reshape(attention2, (32,600)),0.8)
            #attentio2 = rnn.DropoutWrapper(attention, output_keep_prob=0.5)
            #softmax_w_att_2 = self.weight_variable([600, 600])
            #softmax_b_att_2 = self.bias_variable([32, 600])
            #attention4 = tf.matmul(attention, softmax_w_att_2) + softmax_b_att_2
            #self.attention = tf.reshape(attention4,(32,600))
            #self.attention = tf.argmax(attention5, 1, name="att5")

    def data_format(self, _data, _format=tf.float32):
        return tf.cast(_data,dtype=_format)

    def init_cnn(self):
        a=tf.reshape(self.X_inputs,(6400,128))
        _b=tf.reshape(self.attention, [-1])
        __b = tf.expand_dims(_b, -1)
        b=self.data_format(__b, tf.float32)
        c=tf.multiply(a,b)
        if Const.DEBUG == "True":
            pdb.set_trace()
        d = tf.reshape(c,(32,self.timestep_size*self.embedding_size))
        inputy = tf.one_hot(self.y_inputs,18)
        cnn_model = cnn(
            sequence_length=32,
            num_classes=18,
            vocab_size=120000,
            embedding_size=128,
            num_filters=32,
            input_x=d,
            input_y=inputy,
            prob=0.5,
            filter_sizes=[2,3,4],
            l2_reg_lambda=0.0)
        self.classify_pred = cnn_model.predictions
        self.classify_losses = tf.nn.softmax_cross_entropy_with_logits(logits=cnn.scores, labels=self.y_inputs)

    def init_model_struct(self):
        self.input_placeholder()
        self.lr = tf.placeholder(tf.float32, [])
        self.keep_prob = tf.placeholder(tf.float32, [])
        self.batch_size = tf.placeholder(tf.int32, [])  # 注意类型必须为 tf.int32
        #self.init_embedding()
        self.bilstm_output = self.init_bi_lstm()
        #self.init_bi_lstm()
        self.init_outputs()
        self.init_attention()
        self.init_cnn()
        # adding extra statistics to monitor
        # self.y_inputs.shape = [self.batch_size, self.timestep_size]
        #self.correct_prediction = tf.equal(tf.cast(tf.argmax(self.y_pred, 1), tf.int32), tf.reshape(self.y_inputs, [-1]))
        #pdb.set_trace()
        #xx = tf.reshape(self.y_inputs, [-1])
        #xx = tf.cast(xx, dtype=tf.float32)
        #yy = tf.reshape(self.attention, [-1])
        #print(xx,yy,xx.shape,yy.shape)
        #print(xx.get_shape().ndims, yy.get_shape().ndims)
        correct_prediction = tf.equal(tf.cast(tf.argmax(self.attention, 1), tf.int32), tf.reshape(self.y_inputs, [-1]))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        self.cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = tf.reshape(self.y_inputs, [-1]), logits = self.attention))
        self.loss = 0.1*tf.reduce_mean(self.classify_losses) + self.cost
        #self.cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = tf.reshape(self.y_inputs, [-1]), logits = tf.cast(tf.one_hot(tf.argmax(self.attention,1),3), tf.float32)))
        #tmp_y = tf.reshape(tf.cast(tf.one_hot(self.y_inputs,3), tf.int32), [-1])
        #tmp_y_ = tf.reshape(tf.cast(tf.one_hot(tf.argmax(self.attention,1),3), tf.int32),[-1])
        #self.cost = tf.reduce_mean(tf.equal(tmp_y,tmp_y_))
        tvars = tf.trainable_variables()  # 获取模型的所有参数
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), self.max_grad_norm)  # 获取损失函数对于每个参数的梯度
        optimizer = tf.train.AdamOptimizer(learning_rate=self._lr)   # 优化器
        # 梯度下降计算
        #self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = tf.reshape(self.y_inputs, [-1]), logits = tf.cast(tf.reshape(tf.argmax(self.y_pred,1), [-1]), dtype=tf.float32)))
#attention))
        self.train_op = optimizer.apply_gradients(list(zip(grads, tvars)), global_step=tf.contrib.framework.get_or_create_global_step())
        self.save_graph_meta()

    def mod_lr(self,cost):
        sa = 0.5
        if cost-self.cost>0:
            self._lr = self._lr*(1-sa)
        elif cost-self.cost<0:
            self._lr = self._lr*(1+sa)

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
        tf.add_to_collection('model.y_pred', self.y_pred)
        tf.add_to_collection('model.X_inputs',self.X_inputs)
        tf.add_to_collection('model.y_inputs',self.y_inputs)
        tf.add_to_collection('batch_size',self.batch_size)
        tf.add_to_collection('lr', self.lr)
        tf.add_to_collection('keep_prob', self.keep_prob)
        tf.add_to_collection('attention', self.attention)

    def fit_train(self, sess):
        pass
        data_helper = self.datahelper
        #self.att_layer = text_cnn.TextCNN( \
        #    sequence_length=32, \
        #    num_classes=8, \
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
        test_fetches = [self.attention, self.accuracy, self.cost, self.train_op, self.y_pred]
        train_fetches = [self.attention, self.accuracy, self.cost, self.train_op]
        #train_att_fetches = [self.att_layer.accuracy, self.att_layer.train_op, self.att_layer.predictions]
        for epoch in range(self.max_max_epoch):
            #self.datahelper.train_data_generator = self.datahelper.gen_train_data(per=0.8,name='train')
            #start_time = time.time()
            _costs, _accs, show_accs, show_costs  = 0.0, 0.0, 0.0, 0.0
            #_costs, _accs = 0.0, 0.0
            for batch in range(self.tf_batch_num):
                _print('EPOCH %d lr=%g' % (epoch+1, self._lr))
                _acc = 0.0
                self.cost = 0.0
                X_batch, y_batch = self.datahelper.next_batch_text_classify_train(self.gen)
                feed_dict = {self.X_inputs:X_batch, self.y_inputs:y_batch, self.lr:self._lr, self.batch_size:32, self.keep_prob:0.5}
                _att, _acc, _cost, _ = sess.run(train_fetches, feed_dict) # the self.cost is the mean self.cost of one batch
                #att_feed_dict={self.att_layer.input_x:self.textcnn_data_transform(_att,5), self.att_layer.input_y:tf.reshape(tf.one_hot(y_batch,1),(1000,8)), self.att_layer.dropout_keep_prob:0.5 }
                #_att_acc, _att_op, _att_pred, _ = sess.run(train_att_fetches, att_feed_dict) # the self.cost is the mean self.cost of one batch
                #_print(dict(zip(["_att_acc", "_att_op", "_att_pred"],[ _att_acc, _att_op, _att_pred])))
                _print("\n> y_inputs: ",list(y_batch))
                _print("\n> self.attention: ",_att)
                _print("\n> _att :",_att," _acc:",_acc," _cost:",_cost)
                #_accs += _acc
                #_costs += _cost
                show_accs += _acc
                show_costs += _cost
                #_print("show_accs, show_costs, _accs, _costs")
                _print(show_accs/(batch+1), show_costs/(batch+1))
                self.mod_lr(show_costs)
                print("acc cost average")
                if batch%3==1:
                   X_batch, y_batch = self.datahelper.next_batch()
                   feed_dict = {self.X_inputs:X_batch, self.y_inputs:y_batch, self.lr:self._lr, self.batch_size:32, self.keep_prob:0.5}
                   _att, _acc, _cost, _op_, y_pred = sess.run(test_fetches, feed_dict)
                   mean_acc = _acc/batch
                   mean_cost = _cost/batch
                   _print("test acc per sentence, cost per sentence")
                   _print(mean_acc,mean_cost)
                   _print("\n> _att :",_att," _acc:",_acc," _cost:",_cost)

                if batch%100==1:
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
        saver = tf.train.Saver(max_to_keep=30)
        saver.restore(sess, tf.train.latest_checkpoint(self.model.checkpoint_path))
        X_batch, y_batch = datahelper.next_batch('eval')
        test_fetches = [self.attention, self.accuracy_attention, self.accuracy_bilstm, self.y_pred_meta]
        feed_dict = {self.X_inputs:X_batch, self.y_inputs:y_batch, self.lr:self._lr, self.batch_size:32, self.keep_prob:1.0}
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


