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

with open('/home/siyuan/data/data2.pkl', 'rb') as inp:
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

# adding extra statistics to monitor
# y_inputs.shape = [batch_size, timestep_size]

# ***** 优化求解 *******

# 梯度下降计算

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

import myconfig as config

def classifier_words(dat, pred):
    dat = dat[:len(pred)]
    cnt = 0
    flag = False
    #context = "".join(dat)
    context = dat
    _words = list(jieba.cut(context, HMM=True))
    words = _words
    cls = []
    mark = ""
    for word in words:
        _d = {}
        print("\n\n word %s"%word)
        _l  =len(word)
        for j in pred[cnt : cnt + _l]:
            for p in config.key_name_lst:
                if j in config.key_dict[p]:
                    if p in _d.keys():
                        _d[p] += 1
                    else:
                        _d[p] = 1
        _f = dict(zip(_d.values(), _d.keys()))
        cls_ = _f[max(_f)]
        cls.append(cls_)
        print(word, "is", cls_)
        print("\n> %s : %s "% (word, "".join(pred[cnt : cnt + _l])))
        cnt += _l
    #print(len(words))
    #print(len(cls))
    print("\n>words %s" % ", ".join(words))
    print("\n>cls %s" % ", ".join(cls))
    assert len(words) == len(cls)
    return words, cls
tr_batch_size = 256
#max_max_epoch = 6
display_num = 500  # 每个 epoch 显示是个结果

sess.run(tf.global_variables_initializer())

new_saver=tf.train.import_meta_graph('/home/siyuan/data/model/momo/bilstm.ckpt-8.meta')
new_saver.restore(sess,'/home/siyuan/data/model/momo/bilstm.ckpt-8')
graph = tf.get_default_graph()
X_inputs=tf.get_collection("X_inputs")[0]
y_inputs=tf.get_collection("y_inputs")[0]
y_pred_meta=tf.get_collection("y_pred")[0]
lr=tf.get_collection("lr")[0]
batch_size=tf.get_collection("batch_size")[0]
keep_prob=tf.get_collection("keep_prob")[0]

for epoch in range(max_max_epoch):
    _lr = 1e-4
    #print('EPOCH %d， lr=%g' % (epoch+1, _lr))
    start_time = time.time()
    _costs = 0.0
    _accs = 0.0
    show_accs = 0.0
    show_costs = 0.0
    #print("y_inputs:y_batch")
    for batch in range(1): 
        X_batch, y_batch = data_train.next_batch(1)
        X_batch = [word2id[i] for i in X_batch]
        y_batch = [tag2id[i] for i in y_batch]
        feed_dict = {X_inputs:X_batch, y_inputs:y_batch, lr:1e-4, batch_size:1.0, keep_prob:1.0}
        #print("y_pred 预测值是:", sess.run(y_pred_meta, feed_dict=feed_dict))
        fetches = [y_pred_meta]
        _y_pred_meta = sess.run(fetches, feed_dict=feed_dict) # the cost is the mean cost of one batch
        print(_y_pred_meta)

print('**TEST RESULT:')

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


#============================================================================================================
#
# block9: eval the model
#
#============================================================================================================

#for i in range(1):
#    print(",".join(datas[i]))
#    print(",".join(labels[i]))
#    print(",".join(simple_cut(datas[i])))


#_lines = ""
#with open("/home/siyuan/data/cut_char_without_marker_3.txt", "r") as f:
#    cont= f.read()
#    lines = cont.split("\n")
#    _lines = lines[10000:100000]

#print(_lines[0])
#print(_lines[10])
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

def simple_cut(text, tags, y_pred):
    """对一个片段text（标点符号把句子划分为多个片段）进行预测。"""
    if len(text)>0:
        text_len = len(text)
        print("\n print the ", text_len, text)
        _y_pred = ""
        try:
            X_batch = text2ids(text)  # 这里每个 batch 是一个样本
            #print("\n> X_batch.shape", X_batch.shape)
            feed_dict = {X_inputs:np.array(X_batch).reshape(1,100), lr:1e-4, batch_size:1.0, keep_prob:1.0}
            _y_pred = sess.run([y_pred], feed_dict)[0][:text_len]  # padding填充的部分直接丢弃
            print("_y_pred")
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
            return []
        nodes = [dict(list(zip(tags[1:], each[1:]))) for each in _y_pred]
        #print("\n> 使用模型训练，预测节点:", nodes)
        vittags = viterbi(nodes)
        for i in range(len(nodes)):
            with open ("nodes.txt", "a+") as f:
                f.write("\n {%s : %s}" % (text[i], str(nodes[i])))
        return vittags
    else:
        return []
zy = ""
with open("/home/siyuan/data/zy_json.json", "r") as f:
    cont = f.read()
    zy = json.loads(cont)
for i in zy.keys():
    zy[i] = zy[i]/5

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

def get_Xy(sentence):
    """将 sentence 处理成 [word1, w2, ..wn], [tag1, t2, ...tn]"""
    words_tags = re.findall('(.)/(.)', sentence)
    if words_tags:
        words_tags = np.asarray(words_tags)
        words = words_tags[:, 0]
        tags = words_tags[:, 1]
        return words, tags # 所有的字和tag分别存为 data / label
    return None

def clr(line):
    line = re.sub("[^\u4e00-\u9fa50-9a-zA-Z]","",line)
    return line

def replaceNC(line):
    line = re.sub("[0-9]","3",line)
    line = re.sub("[A-Pa-p]","c",line)
    line = re.sub("[R-Zr-z]","c",line)
    line = re.sub("(qq|QQ)","QQ",line)
    line = re.sub("[^\u4e00-\u9fa50-9a-zA-Z@]","",line)
    return line

def isweb(line):
    if len(re.findall("(邮箱|网址|网站)",line))>0:
        return True
    else:
        return False

def isnickname(line):
    if len(re.findall("外号叫",line))>0:
        return True
    else:
        return False

def ispass(line):
    if len(re.findall("[\u4e00-\u9fa5]",line))<1:
        return False
    if len(re.findall("\d{7,}",line))>0:
        return True 
    if len(re.findall("外号叫",line))>0:
        return True
    if len(re.findall("微信",line))>0:
        return True
    if len(re.findall("qq",line))>0:
        return True
    if len(re.findall("QQ",line))>0:
        return True
    if len(re.findall("邮箱",line))>0:
        return True
    if len(re.findall("陌陌",line))>0:
        return True
    if len(re.findall("网址",line))>0:
        return True
    if len(re.findall("www",line))>0:
        return True
    if len(re.findall("com",line))>0:
        return True
    return False

def handle_all_data():
    ending = 0
    with open("/home/siyuan/data/pred_momo.txt", 'a+')as nick:
        with open("/home/siyuan/data/ner_sample.txt", "r") as f:
            while(1):
                ending+=1
                #if ending<1700000:
                #    continue
                if ending%10000 ==10:
                    print(ending)
                cont= f.readline()
                cont= re.sub("_","",cont)
                if cont == "":
                    print("context read finish")
                    break
                if not isweb(cont):
                  if not isnickname(cont):
                    continue
               	nick.write("\n\n> text     :"+cont)
                clrtext = replaceNC(cont)
                nick.write("\n> clrtext  :"+clrtext)
                pred = simple_cut(clrtext, config.tags, y_pred_meta)
                if pred == []:
                   print("pred is null")
                   continue
                print(ending)
                print(pred)
                nick.write("\n> pred     :"+"".join(pred))
                words, clss = classifier_words(clrtext, pred)
                for i in range(len(words)):
                    print("\t>{%s:%s}"%(words[i], clss[i]))
                    if clss[i] == "white":
                        continue
                    nick.write("\t>{%s:%s}"%(words[i], clss[i]))
handle_all_data()
print("handle_all_data() finish")

'''
while(1):
    pass

with open ("pred_bilstm_ext.txt", "a+") as fo:
  cnt = 0
  for line in _lines:
    if len(re.findall("外/(.) 号/(.)", line)) <1:
        continue
    print(line)
    """""
    if not "银行" in line:
        continue
    if not "车牌" in line:
        continue
    if not "陌陌" in line:
        continue
    if not "微信" in line:
        continue
    if not "QQ" in line:
        continue
    """

    cnt+=1
    if cnt%10==1:
        print("\n> this is the %d one"%cnt )
    dat = 0
    lab = 0
    try:
        dat, lab = get_Xy(line) 
    except:
        print(line)
        traceback.print_exc()
        continue
        
    fo.write("\n\n> text     :"+"".join(dat))
    fo.write("\n> label    :"+"".join(lab))
    pred = simple_cut(''.join(dat), config.tags, y_pred_meta)
    fo.write("\n> pred     :"+"".join(pred))
    words, clss = classifier_words(''.join(dat), pred)

    for i in range(len(words)):
        if clss[i] == "white":
            continue
        fo.write("\n>{%s:%s}"%(words[i], clss[i]))

#============================================================================================================
#
# end
#
#============================================================================================================
'''
