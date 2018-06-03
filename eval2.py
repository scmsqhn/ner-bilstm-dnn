# eval2.py

import json
import sys
sys.path.append("/home/siyuan/svn/algor/src/iba/dmp/gongan/storm_crim_classify/extcode")
sys.path.append("/home/siyuan/svn/algor/src/iba/dmp/gongan/storm_crim_classify/extcode")
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
import config

DEBUG =True
DATA = True
import json

tags = ['z','b','i','e','s','p','h','n','u','v','x','d','t','f','Q','q','k','c','r','R']

white  = ['z','b','i','e','s']
phone  = ['p','h','n']
wx     = ['u','v','x']
iden   = ['d','t','f']
qq     = ['Q','q', 'k']
credit = ['c','r','R']

cls_lst = {}
cls_lst["white"] = white
cls_lst["phone"] = phone
cls_lst["wx"] = wx
cls_lst["iden"] = iden
cls_lst["qq"] = qq
cls_lst["credit"] = credit

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

import pickle
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
print(word2id)
#============================================================================================================
#
# end
#
#============================================================================================================
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train,  test_size=0.2, random_state=42)
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

#print('Creating the data generator ...')
#data_train = BatchGenerator(X_train, y_train, shuffle=True)
print("\n> new a BatchGenerator")
#print("\n> data_train.X.shape: ", data_train.X.shape)
#print("\n> data_train.y.shape: ", data_train.y.shape)
#data_valid = BatchGenerator(X_valid, y_valid, shuffle=False)
#data_test = BatchGenerator(X_test, y_test, shuffle=False)
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

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
from tensorflow.contrib import rnn
import numpy as np

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
class_num = len(tags)
hidden_size = 128    # 隐含层节点数
layer_num = 2        # bi-lstm 层数
max_grad_norm = 5.0  # 最大梯度（超过此值的梯度将被裁剪）

# ====================== model inout=====================
lr = tf.placeholder(tf.float32, [])
keep_prob = tf.placeholder(tf.float32, [])
batch_size = tf.placeholder(tf.int32, [])  # 注意类型必须为 tf.int32
model_save_path = './ckpt/bi-lstm.ckpt-1'  # 模型保存位置

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
    
    # 下面两部分是等价的
    # **************************************************************
    # ** 把 inputs 处理成 rnn.static_bidirectional_rnn 的要求形式
    # ** 文档说明
    # inputs: A length T list of inputs, each a tensor of shape
    # [batch_size, input_size], or a nested tuple of such elements.
    # *************************************************************
    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    # inputs.shape = [batchsize, timestep_size, embedding_size]  ->  timestep_size tensor, each_tensor.shape = [batchsize, embedding_size]
    # inputs = tf.unstack(inputs, timestep_size, 1)
    # ** 3.bi-lstm 计算（tf封装）  一般采用下面 static_bidirectional_rnn 函数调用。
    #   但是为了理解计算的细节，所以把后面的这段代码进行展开自己实现了一遍。
#     try:
#         outputs, _, _ = rnn.static_bidirectional_rnn(cell_fw, cell_bw, inputs, 
#                         initial_state_fw = initial_state_fw, initial_state_bw = initial_state_bw, dtype=tf.float32)
#     except Exception: # Old TensorFlow version only returns outputs not states
#         outputs = rnn.static_bidirectional_rnn(cell_fw, cell_bw, inputs, 
#                         initial_state_fw = initial_state_fw, initial_state_bw = initial_state_bw, dtype=tf.float32)
#     output = tf.reshape(tf.concat(outputs, 1), [-1, hidden_size * 2])
    # ***********************************************************
    
    # ***********************************************************
    # ** 3. bi-lstm 计算（展开）
    with tf.variable_scope('bidirectional_rnn'):
        # *** 下面，两个网络是分别计算 output 和 state 
        # Forward direction
        outputs_fw = list()
        state_fw = initial_state_fw
        with tf.variable_scope('fw'):
            for timestep in range(timestep_size):
                if timestep > 0:
                    tf.get_variable_scope().reuse_variables()
                (output_fw, state_fw) = cell_fw(inputs[:, timestep, :], state_fw)
                outputs_fw.append(output_fw)
        
        # backward direction
        outputs_bw = list()
        state_bw = initial_state_bw
        with tf.variable_scope('bw') as bw_scope:
            inputs = tf.reverse(inputs, [1])
            for timestep in range(timestep_size):
                if timestep > 0:
                    tf.get_variable_scope().reuse_variables()
                (output_bw, state_bw) = cell_bw(inputs[:, timestep, :], state_bw)
                outputs_bw.append(output_bw)
        # *** 然后把 output_bw 在 timestep 维度进行翻转
        # outputs_bw.shape = [timestep_size, batch_size, hidden_size]
        outputs_bw = tf.reverse(outputs_bw, [0])
        # 把两个oupputs 拼成 [timestep_size, batch_size, hidden_size*2]
        output = tf.concat([outputs_fw, outputs_bw], 2)
        output = tf.transpose(output, perm=[1,0,2])
        output = tf.reshape(output, [-1, hidden_size*2])
    # ***********************************************************
    return output # [-1, hidden_size*2]


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
train_op = optimizer.apply_gradients( list(zip(grads, tvars)),
    global_step=tf.contrib.framework.get_or_create_global_step())
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
        feed_dict = {X_inputs:X_batch, y_inputs:y_batch, lr:1e-5, batch_size:_batch_size, keep_prob:1.0}
        _acc, _cost = sess.run(fetches, feed_dict)
        _accs += _acc
        _costs += _cost    
    print("\n> batch_num: ", batch_num)
    print("\n> acc 10个字一组，每组acc求和处以组数")
    mean_acc= _accs / batch_num     
    mean_cost = _costs / batch_num
    return mean_acc, mean_cost

"""""
# ===========================================================================
#
# block train run 
#
# ===========================================================================

sess.run(tf.global_variables_initializer())
tr_batch_size = 64 
#max_max_epoch = 6
display_num = 10  # 每个 epoch 显示是个结果
print("\n> data_train", data_train)
tr_batch_num = int(data_train.y.shape[0] / tr_batch_size)  # 每个 epoch 中包含的 batch 数
print("\n> tr_batch_num", tr_batch_num)

display_batch = int(tr_batch_num / display_num)  # 每训练 display_batch 之后输出一次
print("display_batch", display_batch)
saver = tf.train.Saver(max_to_keep=10)  # 最多保存的模型数量
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
        fetches = [accuracy, cost, train_op]
        X_batch, y_batch = data_train.next_batch(tr_batch_size)
        print(y_batch)
        feed_dict = {X_inputs:X_batch, y_inputs:y_batch, lr:_lr, batch_size:tr_batch_size, keep_prob:0.5}
        _acc, _cost, _ = sess.run(fetches, feed_dict) # the cost is the mean cost of one batch
        _accs += _acc
        _costs += _cost
        show_accs += _acc
        show_costs += _cost
        #print(display_batch)
        #print(batch)
        if (batch + 1) % display_batch == 0:
            valid_acc, valid_cost = test_epoch(data_valid)  # valid
            print('\ttraining acc=%g, cost=%g;  valid acc= %g, cost=%g ' % (show_accs / display_batch,
                                                show_costs / display_batch, valid_acc, valid_cost))
            show_accs = 0.0
            show_costs = 0.0
    mean_acc = _accs / tr_batch_num 
    mean_cost = _costs / tr_batch_num
    if (epoch + 1) % 3 == 0:  # 每 3 个 epoch 保存一次模型
        save_path = saver.save(sess, model_save_path, global_step=(epoch+1))
        print('the save path is ', save_path)
    print('\ttraining %d, acc=%g, cost=%g ' % (data_train.y.shape[0], mean_acc, mean_cost))
    print('Epoch training %d, acc=%g, cost=%g, speed=%g s/epoch' % (data_train.y.shape[0], mean_acc, mean_cost, time.time()-start_time))        
# testing
print('**TEST RESULT:')
test_acc, test_cost = test_epoch(data_test)
print('**Test %d, acc=%g, cost=%g' % (data_test.y.shape[0], test_acc, test_cost)) 


# ===========================================================================
#
# block train run 
#
# ===========================================================================
"""

# 模型测试，现在给定一个字符串，首先应该把处理成正确的模型输入形式。即每次输入一个片段，（这里限制了每个片段的长度不超过 max_len=32）。每个字处理为对应的 id， 每个片段都会 padding 处理到固定的长度。也就是说，输入的是一个list， list 的每个元素是一个包含多个 id 的list。<br/>
# 即 [[id0, id1, ..., id31], [id0, id1, ..., id31], [], ...]

#============================================================================================================
#
# block 8: model eval, viterbe 
#
#============================================================================================================

# ** 导入模型
#saver = tf.train.Saver()
#best_model_path = 'ckpt/bi-lstm.ckpt'
#saver.restore(sess, best_model_path)

saver = tf.train.Saver()
saver.restore(sess, model_save_path)

def load_model():
    saver = tf.train.Saver()
    saver.restore(sess, model_save_path)

# 再看看模型的输入数据形式, 我们要进行分词，首先就要把句子转为这样的形式
#X_tt, y_tt = data_train.next_batch(2)
#print('X_tt.shape=', X_tt.shape, 'y_tt.shape=', y_tt.shape)
#print('X_tt = ', X_tt)
#print('y_tt = ', y_tt)

# ## 维特比解码
# 下面使用维特比的方法来计算的最优的状态序列。具体原理可以参考 隐马尔可夫模型。在 HMM 监督学习中，我们统计样本频数来得到转移概率和 
# 利用 labels（即状态序列）来统计转移概率
# 因为状态数比较少，这里用 dict={'I_tI_{t+1}'：p} 来实现
# A统计状态转移的频数

def reverse_dic(d):
    _d = {}
    for i in d.keys():
        _d[d[i]] = i
    return _d


def zy_mat(label):
    A = {}
    B = {}
    status_lst = tags[1:]
    for i in status_lst:
        for j in status_lst:
            A["%s%s"%(i,j)] = 1e-9 
        B[i] = 1e-9

    zy = dict()
    for t in range(len(label) - 1):
            key = label[t] + label[t+1]
            A[key] += 1.0
            B[label[t]] += 1.0
        
    zy = {}
    zy_keys = list(set(A.keys()))
    for key in zy_keys:
        zy[key] = A[key] / B[key[1]]
        print(zy[key])
        print(A[key])
        print(B[key[1]])

    keys = sorted(zy.keys())
    print('the transition probability: ')
    for key in keys:
        print(key, zy[key])
    
    zy = {i:zy[i]*5 for i in list(zy.keys())}
    #zy = {i:np.log(zy[i]+1.0) for i in list(zy.keys())}
    return zy
zy = ""
sav_lod = False #True
#sav_lod = True

if sav_lod:
   y_str=""
   ll = list()
   [ll.append(i) for i in range(len(tags))]
   _d = dict(zip(ll, tags))
   for i in y:
       for j in i:
           y_str+=_d[j]
   y_str = re.sub("z","",y_str)
   print("\n> the str label y_str:", y_str)
   zy = zy_mat(y_str)

   with open("zy_json.json", "w+") as f:
       f.write(json.dumps(zy))
       print("\n> zy_json write 2 json over")
else:
   with open("zy_json.json", "r") as f:
        cont = f.read()
        zy = json.loads(cont)

def viterbi(nodes):
    #print("viterbi nodes list")
    #paths = {'b': nodes[0]['b']} # 第一层，只有一个节点
    paths = {'b': nodes[0]['b'], 's':nodes[0]['s']} # 第一层，只有两个节点
    for layer in range(1, len(nodes)):  # 后面的每一层
        log("nodes 一共有多少层")
        log(len(nodes))
        paths_ = paths.copy()  # 先保存上一层的路径
        log("保存每一层")
        log(paths_)
        # node_now 为本层节点， node_last 为上层节点
        paths = {}  # 清空 path 
        for node_now in list(nodes[layer].keys()):
            # 对于本层的每个节点，找出最短路径
            sub_paths = {} 
            # 上一层的每个节点到本层节点的连接
            for path_last in list(paths_.keys()):
                if path_last[-1] + node_now in list(zy.keys()): # 若转移概率不为 0 
                    sub_paths[path_last + node_now] = paths_[path_last] + nodes[layer][node_now] + zy[path_last[-1] + node_now]
                    pass#print(sub_paths)
            # 最短路径,即概率最大的那个
            sr_subpaths = pd.Series(sub_paths)
            sr_subpaths = sr_subpaths.sort_values()  # 升序排序
            node_subpath = sr_subpaths.index[-1]  # 最短路径
            node_value = sr_subpaths[-1]   # 最短路径对应的值
            # 把 node_now 的最短路径添加到 paths 中
            paths[node_subpath] = node_value
    # 所有层求完后，找出最后一层中各个节点的路径最短的路径
    sr_paths = pd.Series(paths)
    sr_paths = sr_paths.sort_values()  # 按照升序排序
    #print("viterbi返回的排列")
    #print(sr_paths.index[-1])
    return sr_paths.index[-1]  # 返回最短路径（概率值最大的路径）

def text2ids(text):
    """把字片段text转为 ids."""
    words = list(text)
    ids = list(word2id[words])
    if len(ids) >= max_len:  # 长则弃掉
        print('输出片段超过%d部分无法处理' % (max_len)) 
        return ids[:max_len]
    ids.extend([0]*(max_len-len(ids))) # 短则补全
    ids = np.asarray(ids).reshape([-1, max_len])
    return ids

def compare_predict(tags, predict, text):
    return pd.DataFrame([tags,predict, text])

def simple_cut(text):
    """对一个片段text（标点符号把句子划分为多个片段）进行预测。"""
    tags = ['z','b','i','e','s','p','h','n','u','v','x','d','t','f','Q','q','k','c','r','R']
    if len(text)>0:
        text_len = len(text)
        X_batch = text2ids(text)  # 这里每个 batch 是一个样本
        fetches = y_pred
        feed_dict = {X_inputs:np.array(X_batch).reshape(-1,100), lr:1.0, batch_size:1, keep_prob:1.0}
        _y_pred = sess.run([fetches], feed_dict)[0][:text_len]  # padding填充的部分直接丢弃
        #nodes = [dict(list(zip(['x', 'w','i','n','s','b','m','e'], each))) for each in _y_pred]
        nodes = [dict(list(zip(tags[1:], each[1:]))) for each in _y_pred]
        #print("\n> 使用模型训练，预测节点:", nodes)
        with open("nodes.txt", "a+") as p:
            for i in range(len(text)):
               if i>99:
                   break
               p.write("\n> text   :  "+text[i])
               p.write("\n> nodes  :  "+str(nodes[i]))
        tags = viterbi(nodes)
        return tags
    else:
        return []

def cut_word_base(sentence):
    """首先将一个sentence根据标点和英文符号/字符串划分成多个片段text，然后对每一个片段分词。"""
    #not_cuts = re.compile('([0-9\da-zA-Z ]+)|[。，、？！.\.\?,!]')
    not_cuts = re.compile('\,')
    result = []
    start = 0
    for seg_sign in not_cuts.finditer(sentence):
        result.extend(simple_cut(sentence[start:seg_sign.start()]))
        result.append(sentence[seg_sign.start():seg_sign.end()])
        start = seg_sign.end()
    log("测试用的分词结果　result.extend(simple_cut(sentence[start:]))")
    result.extend(simple_cut(sentence[start:]))
    log(result)
    return result

def cut_word(sentence):
    """首先将一个sentence根据标点和英文符号/字符串划分成多个片段text，然后对每一个片段分词。"""
    result = simple_cut(sentence)
    return result

def evalueate(sentence):
    #sentence = '我的微信号码是wxid13678028750'
    result = cut_word(replaceNC(sentence))
    rss = ''
    for each in result:
        rss = rss + each + ' / '
        print(rss)

"""
with open("/home/siyuan/data/beijing110_cp.txt", "r") as f:
    cnt = 10000
    while(cnt>0):
        line = f.readline()
        cont = "".join(list(jieba.cut(line, HMM=True)))
        evalueate(clear_marker(cont))
        cnt-=1
"""
def wx_data_init():
    #clstxt.prepare_train_eval()
    with open("~/gensim_word2vec/train", "r") as f:
        train = f.readlines()
    with open("~/gensim_word2vec/evalu", "r") as g:
        evalu = g.readlines()
    wxLst = list()
    js_train = digiext.extract_digital(train)
    js_evalu = digiext.extract_digital(evalu)
    for i in js_train:
        wxLst.append(js_train[i]['wx'])
    for i in js_evalu:
        wxLst.append(js_evalu[i]['wx'])
    wxLst = list(set("".join(exLst).split(",")))
    print(wxLst[:3])
    for i in js_train:
        data_prepare("train_wx" ,js_train[i]['content'])
    for j in js_evalu:
        data_prepare("evalu_wx" ,js_evalu[i]['content'])

def phone_data_init():
    #clstxt.prepare_train_eval()
    with open("/home/siyuan/data/beijing110_cont.txt", "r") as f:
        cont = f.read()
    formula_cont = data_helper.formula_text(cont)
    mark_the_phonenum("/home/siyuan/data/beijing_phong_ext_train.txt", formula_cont)


#============================================================================================================
#
# block9: eval the model
#
#============================================================================================================

white  = ['z','b','i','e','s']
phone = ['p','h','n']
wx = ['u','v','x']
iden = ['d','t','f']
qq = ['Q','q', 'k']
credit = ['c','r','R']

def clasifier_word(pred):
    l = len(pred)
    cnt = ""
    if l>3:
        _pred = pred[-3:]
    else:
        _pred = pred

    for i in _pred:
            if i in white:
                if cnt == "white":
                    return "white"
                else:
                   cnt = "white"

            if i in white:
                if cnt == "phone":
                    return "phone"
                else:
                   cnt = "phone"

            if i in white:
                if cnt == "wx":
                    return "wx"
                else:
                   cnt = "wx"

            if i in white:
                if cnt == "qq":
                    return "qq"
                else:
                   cnt = "qq"

            if i in white:
                if cnt == "iden":
                    return "iden"
                else:
                   cnt = "iden"
            if i in white:
                if cnt == "credit":
                    return "credit"
                else:
                   cnt = "credit"
            return "white"

#import digital_extract_classfier_regre_mul as digimul

def ret_cls(char):
    for j in cls_lst:
        if char in j:
            return str(j)
    

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



             


    pass

def get_Xy(sentence):
    """将 sentence 处理成 [word1, w2, ..wn], [tag1, t2, ...tn]"""
    words_tags = re.findall('(.)/(.)', sentence)
    if words_tags:
        words_tags = np.asarray(words_tags)
        words = words_tags[:, 0]
        tags = words_tags[:, 1]
        cnt =  0
        for i in range(len(words_tags)):
            #print(tags[i])
            if tags[i] in "bmeswin":
                continue
            #print(words[i])
            #print(tags[i])
        log(words)
        log(tags)
        return words, tags # 所有的字和tag分别存为 data / label
    log("没有标签文档 返回空")
    return None

lines = list()
with open("cut_char_without_marker.txt", "r") as f:
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

    #fo.write("\n white  = ['z','b','i','e','s'], "\
    #"phone = ['p','h','n'], "\
    #"wx = ['u','v','x'], "\
    #"iden = ['d','t','f'], "\
    #"qq = ['Q','q', 'k'], "\
    #"credit = ['c','r','R'], ")

fo.close()

#============================================================================================================
#
# end
#
#============================================================================================================
