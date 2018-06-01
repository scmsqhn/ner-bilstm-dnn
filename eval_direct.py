import myconfig as config
import re
import jieba
import traceback
import pickle
import json
import numpy as np
import pandas as pd
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

def maxone(data_value):
    print(data_value[0])
    print(data_value[-1])
    data_shape = data_value.shape
    data_rows = data_shape[0]
    data_cols = data_shape[1]
    #print(data_shape[0])
    #print(data_shape[1])
    #print(len(data_value[0]))
    for i in range(0, data_rows):
        m = max(data_value[i])
        for j in range(0, data_cols):
            print("\n",data_value[i][j])
            if m == data_value[i][j]:
                continue
            else:
                data_value[i][j] = 0
            print(data_value[i][j])
    print(data_value[0])
    print(data_value[-1])
    return data_value



def standard_deviation_normalization(data_value):
    print(data_value[0])
    print(data_value[-1])
    data_shape = data_value.shape
    data_rows = data_shape[0]
    data_cols = data_shape[1]
    #print(data_shape[0])
    #print(data_shape[1])
    #print(len(data_value[0]))
    for i in range(0, data_rows):
        m = np.average(data_value[i])
        s = np.std(data_value[i])
        for j in range(0, data_cols):
            print("\n",data_value[i][j])
            data_value[i][j] = (data_value[i][j] - m)/s
            print(data_value[i][j])
    print(data_value[0])
    print(data_value[-1])
    return data_value


max_len = 100
zy =  {}

with open("/home/siyuan/data/zy_json.json", "r") as f:
    cont = f.read()
    zy = json.loads(cont)

for i in zy.keys():
    zy[i] = 1.0#zy[i]/3

with open('/home/siyuan/data/data.pkl', 'rb') as inp:
    X = pickle.load(inp)
    y = pickle.load(inp)
    word2id = pickle.load(inp)
    id2word = pickle.load(inp)
    tag2id = pickle.load(inp)
    id2tag = pickle.load(inp)

def text2ids(text):
    """把字片段text转为 ids."""
    words = list(text)
    print(words)
    ids = list(word2id[words])
    print(ids)
    if len(ids) >= max_len:  # 长则弃掉
        print('输出片段超过%d部分无法处理' % (max_len))
        return ids[:max_len]
    ids.extend([0]*(max_len-len(ids))) # 短则补全
    ids = np.asarray(ids).reshape([-1, max_len])
    print("ids")
    print(ids)
    return ids

def viterbi(nodes):
    #print("viterbi nodes list")
    #paths = {'b': nodes[0]['b']} # 第一层，只有一个节点
    paths = {'b': nodes[0]['b'], 's':nodes[0]['s']} # 第一层，只有两个节点
    for layer in range(1, len(nodes)):  # 后面的每一层
        paths_ = paths.copy()  # 先保存上一层的路径
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

def simple_cut(text, tags, y_pred):
    """对一个片段text（标点符号把句子划分为多个片段）进行预测。"""
    if len(text)>0:
        text_len = len(text)
        print("\n print the ", text_len, text)
        _y_pred = ""
        try:
            X_batch = text2ids(text)  # 这里每个 batch 是一个样本
            #print("\n> X_batch.shape", X_batch.shape)
            feed_dict = {X_inputs:np.array(X_batch).reshape(1,100), lr:1e-5, batch_size:1, keep_prob:1.0}
            _y_pred = sess.run([y_pred], feed_dict)[0][:text_len]  # padding填充的部分直接丢弃
            print(_y_pred)
            #_y_pred = standard_deviation_normalization(_y_pred)
            #_y_pred = maxone(_y_pred)
            print(_y_pred.shape)
            print(_y_pred[0])
            print(_y_pred[-1])
            #nodes = [dict(list(zip(['x', 'w','i','n','s','b','m','e'], each))) for each in _y_pred]
        except:
            print("sth is wrong")
            traceback.print_exc()
            return -1
        nodes = [dict(list(zip(tags[1:], each[1:]))) for each in _y_pred]
        #print("\n> 使用模型训练，预测节点:", nodes)
        vittags = viterbi(nodes)
        for i in range(len(nodes)):
            with open ("nodes.txt", "a+") as f:
                f.write("\n {%s : %s}" % (text[i], str(nodes[i])))
        return vittags
    else:
        return []

def classifier_word(pred):
    l = len(pred)
    cnt = ""
    if l>3:
        _pred = pred[-3:]
    else:
        _pred = pred
    for i in _pred:
        key_dict = dict(zip(config.key_name_lst, config.key_value_lst))
        for k in config.key_dict.keys():
            if i in config.key_dict[k]:
                if cnt == k:
                    return k
                else:
                   cnt = k



tfconfig = tf.ConfigProto()
tfconfig.gpu_options.allow_growth = True
from tensorflow.python.framework import ops

ops.reset_default_graph()
sess = tf.InteractiveSession()
#sess = tf.Session(config=tfconfig)


# ====================== config =====================

decay = 0.85
max_epoch = 5
max_max_epoch = 10
timestep_size = max_len = 100           # 句子长度
vocab_size = 6500    # 样本中不同字的个数+1(padding 0)，根据处理数据的时候得到
input_size = embedding_size = 64       # 字向量长度
class_num = len(config.tags)
hidden_size = 512    # 隐含层节点数
layer_num = 2        # bi-lstm 层数
max_grad_norm = 5.0  # 最大梯度（超过此值的梯度将被裁剪）

# ====================== model inout=====================
lr = tf.placeholder(tf.float32, [])
keep_prob = tf.placeholder(tf.float32, [])
batch_size = tf.placeholder(tf.int32, [])  # 注意类型必须为 tf.int32


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

#saver = load_saver(sess) #tf.train.Saver(max_to_keep=10)  # 最多保存的模型数量

with tf.variable_scope('embedding'):
    embedding = tf.get_variable("embedding", [vocab_size, embedding_size], dtype=tf.float32)

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
display_num = 500  # 每个 epoch 显示是个结果
#tr_batch_num = int(data_train.y.shape[0] / tr_batch_size)  # 每个 epoch 中包含的 batch 数
#display_batch = int(tr_batch_num / display_num)  # 每训练 display_batch 之后输出一次
mycnt = 0

#saver = tf.train.Saver(allow_empty=True)
#saver = tf.train.import_meta_graph('/home/siyuan/data/model/ckpy-9.meta')
#saver.restore(sess,ckpt)
graph = tf.Graph()

init_op = tf.global_variables_initializer()
sess.run(init_op)
#saver = tf.train.Saver(max_to_keep=10)
saver = tf.train.import_meta_graph('/home/siyuan/data/model/ckpt-1.meta')

saver.restore(sess, "/home/siyuan/data/model/ckpt-1")
# 通过张量的名称来获取张量
#print(sess.run(tf.get_default_graph().get_tensor_by_name("v1:0")))
#for op in graph.get_operations():
#    print(op.name)
#while(1):
#    print("no op name")
#    pass
#y_pred = graph.get_tensor_by_name('outputs/Variable_1/read:0')

"""""
b_out = self.graph.get_tensor_by_name('b:0')
_input = self.graph.get_tensor_by_name('x:0')
_out = self.graph.get_tensor_by_name('y:0')
y_pre_cls = self.graph.get_tensor_by_name('output:0')
"""
#saver.restore(sess, "/home/siyuan/data/model/ckpt-1")
#saver.restore(sess, '/home/siyuan/data/model/ckpt-1')

#tf.initialize_variables([y_pred])
#saver.restore(sess, tf.train.latest_checkpoint('/home/siyuan/data/model'))
#print("\n new a model from file")
#graph = tf.get_default_graph()
#y_pred = graph.get_tensor_by_name("y_pred:0")
#print("\n WE RESTORED THE Y_PRED FROM META!")

q = open("/home/siyuan/data/shandong_110.txt", "r")
f = open("/home/siyuan/data/pred_eval2.txt", "a+")

print("\n open all data for formula")

cont = q.read()
lines = cont.split("\n")[10000:]
print("\n total has %d sentences here"%len(lines))
for sentence in lines:
    #if not "邮箱" in sentence:
    #    if not "外号" in sentence:
    #        continue
    print("\n ", sentence)
    sentence = re.sub("[^\u4e00-\u9fa5a-zA-Z0-9]","",sentence)
    sentence = re.sub("[0-9]","3",sentence)
    sentence = re.sub("[a-pA-p]","c",sentence)
    sentence = re.sub("[r-zR-Z]","c",sentence)
    sentence = re.sub("(qq|QQ|Qq)","QQ",sentence)
    f.write("\n> text     : "+sentence)
    pred = simple_cut(sentence, config.tags, y_pred)


    print(pred)
    if pred == -1:
        continue
    f.write("\n> pred     : "+ "".join(pred))
    words = jieba.cut(sentence, HMM=True)
    count = 0
    f.write("\n") 
    for word in list(words):
        l = len(word)
        mark = classifier_word(pred[count:count+l])
        print(word)
        print(mark)
        count+=l
        f.write("{%s\t%s}, "%(word,mark))

print("\n close the file and exist")
f.close()
q.close()
