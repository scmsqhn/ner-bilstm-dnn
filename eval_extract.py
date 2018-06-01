# eval_extract.py
import numpy as np
import tensorflow as tf
import re
from tensorflow.contrib import rnn

import pandas as pd
from tqdm import tqdm
from itertools import chain

def replaceNC(line):
    line = re.sub("[0-9]","3",line)
    line = re.sub("[A-Z_a-z]","C",line)
    return line
decay = 0.85
max_epoch = 1
max_max_epoch = 1#3
vocab_size = 6500    # 样本中不同字的个数+1(padding 0)，根据处理数据的时候得到
input_size = embedding_size = 64       # 字向量长度
class_num = 7
hidden_size = 128    # 隐含层节点数
layer_num = 2        # bi-lstm 层数
max_grad_norm = 5.0  # 最大梯度（超过此值的梯度将被裁剪）
max_len = 100
lr = tf.placeholder(tf.float32, [])
keep_prob = tf.placeholder(tf.float32, [])
batch_size = tf.placeholder(tf.int32, [])  # 注意类型必须为 tf.int32
model_save_path = 'ckpt/bi-lstm.ckpt'  # 模型保存位置
datas = list()
labels = list()
datas_eval = list()
labels_eval = list()

with open("/home/siyuan/data/beijing_phong_ext_train_marked.txt","r") as f:
    texts = f.read()
    #sentences = re.split('[\r\n]', texts)
    sentences_eval = re.split('[\r\n]', texts)
    #sep = int(len(sentences)*0.9)
    #
    #train_sentences = shuffle(sentences)[:sep]
    #sentences_eval = shuffle(sentences)[sep:]

def get_Xy(sentence):
    """将 sentence 处理成 [word1, w2, ..wn], [tag1, t2, ...tn]"""
    words_tags = re.findall('(.)/(.)', sentence)
    if words_tags:
        words_tags = np.asarray(words_tags)
        words = words_tags[:, 0]
        tags = words_tags[:, 1]
        cnt =  0
        for i in range(len(words_tags)):
            print(tags[i])
            if tags[i] in "bmeswin":
                continue
            print(words[i])
            print(tags[i])
        ##log(words)
        ##log(tags)
        return words, tags # 所有的字和tag分别存为 data / label
    ##log("没有标签文档 返回空")
    return None

for sentence in tqdm(iter(sentences_eval)):
     result = get_Xy(sentence)
     if result:
          datas.append(result[0])
          labels.append(result[1])
df_data = pd.DataFrame({'words': datas, 'tags': labels}, index=list(range(len(datas))))
all_words = list(chain(*df_data['words'].values))

# 2.统计所有 word
sr_allwords = pd.Series(all_words)
sr_allwords = sr_allwords.value_counts()
set_words = sr_allwords.index
set_ids = list(range(1, len(set_words)+1)) # 注意从1开始，因为我们准备把0作为填充值
tags = ['w','i','n','s','b','m','e']
tag_ids = list(range(len(tags)))
print("> tag_ids")
print(tag_ids)

# 3. 构建 words 和 tags 都转为数值 id 的映射（使用 Series 比 dict 更加方便）
word2id = pd.Series(set_ids, index=set_words)
id2word = pd.Series(set_words, index=set_ids)
tag2id = pd.Series(tag_ids, index=tags)
id2tag = pd.Series(tags, index=tag_ids)
vocab_size = len(set_words)

with tf.Session() as sess:
    saver = tf.train.Saver(allow_empty = True)
    #saver = tf.train.Saver()
    best_model_path = 'ckpt/bi-lstm.ckpt-6'
    saver.restore(sess, best_model_path)

def text2ids(text):
    """把字片段text转为 ids."""
    words = list(text)
    ids = list(word2id[words])
    if len(ids) >= max_len:  # 长则弃掉
        print('输出片段超过%d部分无法处理' % (max_len)) 
        return ids[:max_len]
    ids.extend([0]*(max_len-len(ids))) # 短则补全
    ids = np.asarray(ids).reshape([-1, max_len])
    print("ids")
    print(ids)
    return ids
embedding_size = 100
with tf.variable_scope('embedding'):
    embedding = tf.get_variable("embedding", [vocab_size, embedding_size], dtype=tf.float32)
def lstm_cell():
    cell = rnn.LSTMCell(hidden_size, reuse=tf.get_variable_scope().reuse)
    return rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)

def bi_lstm(X_inputs):
    """build the bi-LSTMs network. Return the y_pred"""
    # X_inputs.shape = [batchsize, timestep_size]  ->  inputs.shape = [batchsize, timestep_size, embedding_size]
    #log("embedding")
    #log(embedding)
    #log(X_inputs)
    #log("X_inputs")
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
    print(output.shape)
    return output # [-1, hidden_size*2]

timestep_size = max_len = 100           # 句子长度

with tf.variable_scope('Inputs'):
    X_inputs = tf.placeholder(tf.int32, [None, timestep_size], name='X_input')
    y_inputs = tf.placeholder(tf.int32, [None, timestep_size], name='y_input')   

bilstm_output = bi_lstm(X_inputs)

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

with tf.variable_scope('outputs'):
    softmax_w = weight_variable([hidden_size * 2, class_num]) 
    softmax_b = bias_variable([class_num]) 
    y_pred = tf.matmul(bilstm_output, softmax_w) + softmax_b

def simple_cut(text):
    """对一个片段text（标点符号把句子划分为多个片段）进行预测。"""
    if text:
        ##log("待检测的text")
        ##log(text)
        text_len = len(text)
        X_batch = text2ids(text)  # 这里每个 batch 是一个样本
        fetches = [y_pred]
        feed_dict = {X_inputs:X_batch, lr:1.0, batch_size:1, keep_prob:1.0}
        ##log("用于检查的数据　feed_dict")
        ##log(feed_dict)
        _y_pred = sess.run(fetches, feed_dict)[0][:text_len]  # padding填充的部分直接丢弃
        ##log("_y_pred")
        ##log(_y_pred)
        #nodes = [dict(list(zip(['x', 'w','i','n','s','b','m','e'], each))) for each in _y_pred]
        nodes = [dict(list(zip(['w','i','n','s','b','m','e'], each))) for each in _y_pred]
        print("> 使用模型训练，找出节点:", nodes)
        ##log("nodes 模型训练出的节点")
        ##log(nodes)
        tags = viterbi(nodes)
        words = []
        for i in range(len(text)):
            #if tags[i] in ['s', 'b']:
            if tags[i] in ['w']:
                words.append(text[i])
            elif tags[i] in ['n']:
                words[-1] += text[i]
                words.append(text[i])
            else:
                pass
                ##log(text[i])
                ###log(words[-1])
                #words[-1] += text[i]
        ##log("打印 words 仅仅包括　微信名")
        ##log(words)
        return words
    else:
        return []

def cut_word(sentence):
    """首先将一个sentence根据标点和英文符号/字符串划分成多个片段text，然后对每一个片段分词。"""
    ##log("测试用的分词结果　result.extend(simple_cut(sentence[start:]))")
    result = simple_cut(sentence)
    ##log(result)
    return result
    
def evalueate(sentence):
    #sentence = '我的微信号码是wxid13678028750'
    result = cut_word(replaceNC(sentence))
    rss = ''
    for each in result:
        rss = rss + each + ' / '
        print(rss)
for line in sentences_eval:
    evalueate(line)
