import sys
sys.path.append("/home/siyuan/svn/algor/src/iba/dmp/gongan/storm_crim_classify/extcode")
sys.path.append("/home/siyuan/svn/algor/src/iba/dmp/gongan/storm_crim_classify/extcode")
sys.path.append("/home/distdev/anaconda3/envs/qq/lib/python3.6/site-packages")

import traceback
#import digital_info_extract as dex
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from tqdm import tqdm
import time
import os
import data_helper
import jieba
import collections
sys.path.append("/home/siyuan/gensim_word2vec")
import classifier_text as clstxt
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sklearn.utils
from sklearn.utils import shuffle
import data_helper
#import data_generator
import myconfig as config
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import json
import pickle

DEBUG =True
DATA = True

tags = config.tags

tfconfig = tf.ConfigProto()
tfconfig.gpu_options.allow_growth = True
sess = tf.Session(config=tfconfig)
"""
envs = dict()
with open("envs.json", "r") as f:
    envs = json.loads(f.read())
"""

def log(n):
    with open("./log.txt", "a+") as f:
        f.write("\n> "+str(n))
        f.write("\n")

#============================================================================================================
#
# block 5:  no need to run the code before u can load the data directlly
#           and separate the data to train and eval 
#
#============================================================================================================

with open('/home/siyuan/data/data.pkl', 'rb') as inp:
    X = pickle.load(inp)
    y = pickle.load(inp)
    #get_ipython().magic('time X = pickle.load(inp)')
    #get_ipython().magic('time y = pickle.load(inp)')
    word2id = pickle.load(inp)
    id2word = pickle.load(inp)
    tag2id = pickle.load(inp)
    id2tag = pickle.load(inp)
print('** Finished loading the data.')    

"""""
print("\n> X,y start to transform")
X = word2id[list(X)]
y = tag2id[list(y)]
print("\n> X,y transoform ok")
"""
#============================================================================================================
#
# end
#
#============================================================================================================

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.001, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train,  test_size=0.001, random_state=42)
print('X_train.shape={}, y_train.shape={}; \nX_valid.shape={}, y_valid.shape={};\nX_test.shape={}, y_test.shape={}'.format(
    X_train.shape, y_train.shape, X_valid.shape, y_valid.shape, X_test.shape, y_test.shape))


#============================================================================================================
#
# block 6: build the generator
#
#============================================================================================================

class BatchGenerator(object):
    """ Construct a Data generator. The input X, y should be ndarray or list like type.
    
    Example:
        Data_train = BatchGenerator(X=X_train_all, y=y_train_all, shuffle=False)
        Data_test = BatchGenerator(X=X_test_all, y=y_test_all, shuffle=False)
        X = Data_train.X
        y = Data_train.y
        or:
        X_batch, y_batch = Data_train.next_batch(batch_size)
     """ 
    
    def __init__(self, X, y, shuffle=False):
        if type(X) != np.ndarray:
            X = np.asarray(X)
        if type(y) != np.ndarray:
            y = np.asarray(y)
        self._X = X
        self._y = y
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._number_examples = self._X.shape[0]
        self._shuffle = shuffle
        if self._shuffle:
            new_index = np.random.permutation(self._number_examples)
            self._X = self._X[new_index]
            self._y = self._y[new_index]
                
    @property
    def X(self):
        return self._X
    
    @property
    def y(self):
        return self._y
    
    @property
    def num_examples(self):
        return self._number_examples
    
    @property
    def epochs_completed(self):
        return self._epochs_completed
    
    def next_batch(self, batch_size):
        """ Return the next 'batch_size' examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._number_examples:
            # finished epoch
            self._epochs_completed += 1
            # Shuffle the data 
            if self._shuffle:
                new_index = np.random.permutation(self._number_examples)
                self._X = self._X[new_index]
                self._y = self._y[new_index]
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._number_examples
        end = self._index_in_epoch
        return self._X[start:end], self._y[start:end]

print('Creating the data generator ...')
data_train = BatchGenerator(X_train, y_train, shuffle=True)
print("\n> new a BatchGenerator")
print("\n> data_train.X.shape: ", data_train.X.shape)
print("\n> data_train.y.shape: ", data_train.y.shape)
data_valid = BatchGenerator(X_valid, y_valid, shuffle=False)
data_test = BatchGenerator(X_test, y_test, shuffle=False)
print('Finished creating the data generator.')


#============================================================================================================
#
# end
#
#============================================================================================================

#============================================================================================================
#
# block 7: model construct
#          Bi-directional lstm  model
#
#============================================================================================================


'''
For Chinese word segmentation.
'''
# ====================== config =====================
decay = 0.85
max_epoch = 5
max_max_epoch = 10
timestep_size = max_len = 100           # 句子长度
vocab_size = 6500    # 样本中不同字的个数+1(padding 0)，根据处理数据的时候得到
input_size = embedding_size = 64       # 字向量长度
class_num = len(config.tags)
hidden_size = 128# 隐含层节点数
layer_num = 2        # bi-lstm 层数
max_grad_norm = 5.0  # 最大梯度（超过此值的梯度将被裁剪）

# ====================== model inout=====================

lr = tf.placeholder(tf.float32, [])
keep_prob = tf.placeholder(tf.float32, [])
batch_size = tf.placeholder(tf.int32, [])  # 注意类型必须为 tf.int32
model_save_path = '/home/siyuan/data/model/hidden'  # 模型保存位置
best_model_path = model_save_path
checkpoint_filepath = model_save_path

with tf.variable_scope('embedding'):
    embedding = tf.get_variable("embedding", [vocab_size, embedding_size], dtype=tf.float32)

def weight_variable(shape):
    """Create a weight variable with appropriate initialization."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def lstm_cell():
    cell = rnn.LSTMCell(hidden_size, reuse=tf.get_variable_scope().reuse)
    return rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
         
def bi_lstm(X_inputs):
    """build the bi-LSTMs network. Return the y_pred"""
    # X_inputs.shape = [batchsize, timestep_size]  ->  inputs.shape = [batchsize, timestep_size, embedding_size]
    inputs = tf.nn.embedding_lookup(embedding, X_inputs)  
    
    # ** 1.构建前向后向多层 LSTM
    cell_fw = rnn.MultiRNNCell([lstm_cell() for _ in range(layer_num)], state_is_tuple=True)
    cell_bw = rnn.MultiRNNCell([lstm_cell() for _ in range(layer_num)], state_is_tuple=True)
  
    # ** 2.初始状态
    initial_state_fw = cell_fw.zero_state(batch_size, tf.float32)
    initial_state_bw = cell_bw.zero_state(batch_size, tf.float32)  
    
    with tf.variable_scope('bidirectional_rnn'):
        outputs_fw = list()
        state_fw = initial_state_fw
        with tf.variable_scope('fw'):
            for timestep in range(timestep_size):
                if timestep > 0:
                    tf.get_variable_scope().reuse_variables()
                (output_fw, state_fw) = cell_fw(inputs[:, timestep, :], state_fw)
                outputs_fw.append(output_fw)
        outputs_bw = list()
        state_bw = initial_state_bw
        with tf.variable_scope('bw') as bw_scope:
            inputs = tf.reverse(inputs, [1])
            for timestep in range(timestep_size):
                if timestep > 0:
                    tf.get_variable_scope().reuse_variables()
                (output_bw, state_bw) = cell_bw(inputs[:, timestep, :], state_bw)
                outputs_bw.append(output_bw)
        outputs_bw = tf.reverse(outputs_bw, [0])
        output = tf.concat([outputs_fw, outputs_bw], 2)
        output = tf.transpose(output, perm=[1,0,2])
        output = tf.reshape(output, [-1, hidden_size*2])
    return output # [-1, hidden_size*2]

saver = tf.train.Saver(max_to_keep=1)

with tf.variable_scope('Inputs'):
    X_inputs = tf.placeholder(tf.int32, [None, timestep_size], name='X_input')
    y_inputs = tf.placeholder(tf.int32, [None, timestep_size], name='y_input')   
    
bilstm_output = bi_lstm(X_inputs)

with tf.variable_scope('outputs'):
    softmax_w = weight_variable([hidden_size * 2, class_num]) 
    softmax_b = bias_variable([class_num]) 
    y_pred = tf.matmul(bilstm_output, softmax_w) + softmax_b
# adding extra statistics to monitor
# y_inputs.shape = [batch_size, timestep_size]
correct_prediction = tf.equal(tf.cast(tf.argmax(y_pred, 1), tf.int32), tf.reshape(y_inputs, [-1]))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = tf.reshape(y_inputs, [-1]), logits = y_pred))

# ***** 优化求解 *******
tvars = tf.trainable_variables()  # 获取模型的所有参数
grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), max_grad_norm)  # 获取损失函数对于每个参数的梯度
optimizer = tf.train.AdamOptimizer(learning_rate=lr)   # 优化器

# 梯度下降计算
train_op = optimizer.apply_gradients(list(zip(grads, tvars)), global_step=tf.contrib.framework.get_or_create_global_step())
print('Finished creating the bi-lstm model.')

# ### 模型训练

def test_epoch(dataset):
    """Testing or valid."""
    _batch_size = 10
    fetches = [accuracy, cost]
    _y = dataset.y
    data_size = _y.shape[0]
    batch_num = int(data_size / _batch_size)
    start_time = time.time()
    _costs = 0.0
    _accs = 0.0
    for i in range(batch_num):
        X_batch, y_batch = dataset.next_batch(_batch_size)
        X_batch = [word2id[i] for i in X_batch]
        y_batch = [tag2id[i] for i in y_batch]
        feed_dict = {X_inputs:X_batch, y_inputs:y_batch, lr:1e-5, batch_size:_batch_size, keep_prob:1.0}
        _acc, _cost = sess.run(fetches, feed_dict)
        _accs += _acc
        _costs += _cost    
    print("\n> batch_num: ", batch_num)
    print("\n> acc 10个字一组，每组acc求和处以组数")
    mean_acc= _accs / batch_num     
    mean_cost = _costs / batch_num
    return mean_acc, mean_cost


tr_batch_size = 256
#max_max_epoch = 6
display_num = 500  # 每个 epoch 显示是个结果
print("\n> data_train", data_train)
tr_batch_num = int(data_train.y.shape[0] / tr_batch_size)  # 每个 epoch 中包含的 batch 数
print("\n> tr_batch_num", tr_batch_num)

display_batch = int(tr_batch_num / display_num)  # 每训练 display_batch 之后输出一次
print("display_batch", display_batch)
mycnt = 0


def load_model(saver):
    try:
        print("\n> load_model")
        ckpt_file = '/home/siyuan/data/model/ckpt'
        checkpoint = tf.train.get_checkpoint_state(ckpt_file)
        print("\n> checkpoint", checkpoint)
        model_file = checkpoint.model_checkpoint_path
        print("\n> model_file", model_file)
        saver.restore(sess, model_file)
        return saver
    except:
        traceback.print_exc()

sess.run(tf.global_variables_initializer())

new_saver=tf.train.import_meta_graph('/home/siyuan/data/model/hidden/bilstm.ckpt-5.meta')
new_saver.restore(sess,'/home/siyuan/data/model/hidden/bilstm.ckpt-5')
graph = tf.get_default_graph()
X_inputs=tf.get_collection("X_inputs")[0]
y_inputs=tf.get_collection("y_inputs")[0]
y_pred_meta=tf.get_collection("y_pred")[0]

for epoch in range(max_max_epoch):
    _lr = 1e-4
    if epoch > max_epoch:
        _lr = _lr * ((decay) ** (epoch - max_epoch))
    #print('EPOCH %d， lr=%g' % (epoch+1, _lr))
    start_time = time.time()
    _costs = 0.0
    _accs = 0.0
    show_accs = 0.0
    show_costs = 0.0
    #print("y_inputs:y_batch")
    for batch in range(tr_batch_num): 
        X_batch, y_batch = data_train.next_batch(tr_batch_size)
        X_batch = [word2id[i] for i in X_batch]
        y_batch = [tag2id[i] for i in y_batch]
        feed_dict = {X_inputs:X_batch, y_inputs:y_batch, lr:1e-5, batch_size:tr_batch_size, keep_prob:0.1}
        #print("y_pred 预测值是:", sess.run(y_pred_meta, feed_dict=feed_dict))
        fetches = [accuracy, cost, y_pred_meta]
        #print("\ntrans")
        #print("\tform")
        #print(y_batch)
        #feed_dict = {X_inputs:X_batch, y_inputs:y_batch, lr:_lr, batch_size:tr_batch_size, keep_prob:0.5}
        #print(feed_dict)
        _acc, _cost, _y_pred_meta = sess.run(fetches, feed_dict=feed_dict) # the cost is the mean cost of one batch
        print(_y_pred_meta)
        print(_acc)
        print(_cost)
        _accs += _acc
        _costs += _cost
        show_accs += _acc
        show_costs += _cost
        #print(display_batch)
        print("\n> %s of %s" % (batch, tr_batch_num))

        #if (batch + 1) % display_batch == 3:
        #    path_ = saver.restore(sess, tf.train.latest_checkpoint(model_save_path))
        #    print(path_,tf.train.latest_checkpoint(model_save_path))

        if (batch + 1) % display_batch == 3:
            pass
            #path_ = saver.restore(sess, tf.train.latest_checkpoint(model_save_path))
            #print(path_,tf.train.latest_checkpoint(model_save_path))
            #x=graph.get_operation_by_name('x').outputs[0]

        if (batch + 1) % display_batch == 8:
            valid_acc, valid_cost = test_epoch(data_valid)  # valid
            print('\ttraining acc=%g, cost=%g;  valid acc= %g, cost=%g ' % (show_accs / display_batch,
                                                show_costs / display_batch, valid_acc, valid_cost))
            show_accs = 0.0
            show_costs = 0.0
    mean_acc = _accs / tr_batch_num 
    mean_cost = _costs / tr_batch_num
    print('\ttraining %d, acc=%g, cost=%g ' % (data_train.y.shape[0], mean_acc, mean_cost))
    print('Epoch training %d, acc=%g, cost=%g, speed=%g s/epoch' % (data_train.y.shape[0], mean_acc, mean_cost, time.time()-start_time))        
# testing
print('**TEST RESULT:')
test_acc, test_cost = test_epoch(data_test)
print('**Test %d, acc=%g, cost=%g' % (data_test.y.shape[0], test_acc, test_cost)) 


# 模型测试，现在给定一个字符串，首先应该把处理成正确的模型输入形式。即每次输入一个片段，（这里限制了每个片段的长度不超过 max_len=32）。每个字处理为对应的 id， 每个片段都会 padding 处理到固定的长度。也就是说，输入的是一个list， list 的每个元素是一个包含多个 id 的list。<br/>
# 即 [[id0, id1, ..., id31], [id0, id1, ..., id31], [], ...]

# In[17]:

#============================================================================================================
#
# block 8: model eval, viterbe 
#
#============================================================================================================

def compare_predict(tags, predict, text):
    return pd.DataFrame([tags,predict, text])

def classifier_words(dat, lab, pred):
    dat = dat[:len(pred)]
    lab = lab[:len(pred)]
    cnt = 0
    flag = False
    context = "".join(dat)
    _words = list(jieba.cut(context, HMM=True))
    words = _words
    words = []
    cls = []
    mark = ""
    for word in words:
        _d = {}
        print("\n\n word %s"%word)
        _l  =len(word)
        for j in pred[cnt : cnt + _l]:
            for p in cls_lst.keys():
                if j in cls_lst[p]:
                    if p in _d.keys():
                        _d[p] += 1
                    else:
                        _d[p] = 1
        cls.append(max(_d))
        print("\n> %s : %s "% (word, "".join(pred[cnt : cnt + _l])))
        cnt += _l
    #print(len(words))
    #print(len(cls))
    #print("\n> %s" % ", ".join(words))
    #print("\n> %s" % ", ".join(cls))
    assert len(words) == len(cls)
    return words, cls

#============================================================================================================
#
# block9: eval the model
#
#============================================================================================================

#for i in range(1):
#    print(",".join(datas[i]))
#    print(",".join(labels[i]))
#    print(",".join(simple_cut(datas[i])))

lines = list()
with open("cut_char_without_marker_3.txt", "r") as f:
    lines = f.readlines()

fo = open("pred_bilstm_ext.txt", "a+")
cnt = 0
for line in lines:
    if not "微" in line:
        continue
    if not "信" in line:
        continue
    cnt+=1
    if cnt%1000==1:
        print("\n> this is the %d one"%cnt )
    dat, lab =  get_Xy(line)
    fo.write("\n\n> text     :"+"".join(dat))
    fo.write("\n> label    :"+"".join(lab))
    pred = simple_cut(''.join(dat))
    words, clss = classifier_words(dat, lab, pred)
    fo.write("\n> pred     :"+"".join(pred))

    for i in range(len(words)):
        if words[i] == "white":
            continue
        else:
            fo.write("\n>{%s:%s}"%(words[i], clss[i]))

fo.close()


#============================================================================================================
#
# end
#
#============================================================================================================
