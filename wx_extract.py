
# coding: utf-8

# # 基于 Bi-directional LSTM 的序列标注任务（分词）
# 
# 
# **tensorflow 版本： 1.2.1**
# 
# 
# 本例子主要参考[【中文分词系列】 4. 基于双向LSTM的seq2seq字标注]{url: http://spaces.ac.cn/archives/3924/} 这篇文章。<br/>
# 该文章用的是 keras 实现的双端 LSTM，在本例中，实现思路和该文章基本上一样，只是用 TensorFlow 来实现的。<br/>
# 
# 本例最主要的是说明基于 TensorFlow 如何来实现 Bi-LSTM。在后面部分进行最后分词处理用的是维特比译码，如果想了解为什么的话可以看一下《统计学习方法》第10章介绍的隐马尔可夫模型。

# 主要参考: <br/>
# [1] 【中文分词系列】 4. 基于双向LSTM的seq2seq字标注 http://spaces.ac.cn/archives/3924/  <br/>
# [2] https://github.com/yongyehuang/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/bidirectional_rnn.py  <br/>
# [3] https://github.com/yongyehuang/deepnlp/blob/master/deepnlp/pos/pos_model_bilstm.py

# In[1]:
import json
import sys
sys.path.append("/home/siyuan/svn/algor/src/iba/dmp/gongan/storm_crim_classify/extcode")
sys.path.append("/home/siyuan/svn/algor/src/iba/dmp/gongan/storm_crim_classify/extcode")
import traceback
import digital_info_extract as dex
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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

rd = dex.RegDict()
##print(rd)
DEBUG =True
DATA = True
# 首先读入语料，将文本中所有的数字和字符进行提取分割和替换，如下
sys.path.append("/home/siyuan/gensim_word2vec")
import classifier_text as clstxt

def log(n):
    with open("./log.txt", "a+") as f:
        f.write("> "+str(n))
        f.write("\n")

def replaceNC(line):
    line = re.sub("[0-9]","3",line)
    line = re.sub("[A-Z_a-z]","C",line)
    return line

#将数字　英文字符进行替换　并提取,分类
#lines = clstxt.load_data()
#lines = [replaceNC(line) for line in lines]

# 将所有的英文数字上下文进行截取，然后聚类分析，标注
def split_by_num_eng(lines):
    print("> split_by_nuym_eng")
    print(len(lines))
    with open("./target_doc4classifier.txt", "w+") as f:
      result = list()
      for line in lines:
        buf = collections.deque(maxlen=5)
        #print(line)
        print(len(lines))
        log("共有多少个buf样本")
        log(len(lines))
        sent = list(jieba.cut(line, HMM=True))
        for i in sent:
            buf.append(i)
            kw = re.findall("[^A-Za-z0-9]", str(buf[-1]))
            print(buf)
            if len(kw)==0:
                log(str(buf[0]))
                result.append("".join(list(buf)))
                log("找到目标字符串,并写入")
                log("".join(list(buf)))
                f.write("".join(list(buf)))
                f.write("\n")
    #log(result)
    return result

#result = split_by_num_eng(lines)
#result = lines
#print(result)
#silhouette_avg, sample_silhouette_values, X, y, lines = clstxt.classifier_doc("./target_doc4classifier.txt", k=20)
#log("完成聚类实现对数字英文的分类")
import digital_info_extract as digiext
"""
if DATA:
    train_data, eval_data = clstxt.prepare_train_eval()
    #mark_train_lines = digiext.extract_digital(train_data)

    log("使用规则　对内容进行标注，后拆分成词段")
"""
def trans_char_num(cut_data):
    cut_ = re.sub("[0-9]","3",cut_data)
    cut_ = re.sub("[a-zA-Z-_]","C",cut_)
    return cut_

def data_prepare_word(filename, lines):
    pass
    texts = ""
    for line in lines:
        texts+=str(line)
        texts+="\n"
    print(texts)
    texts_cut = " ".join(list(jieba.cut(texts, HMM=True)))
    print(texts_cut)
    log(texts_cut)
    log("texts_cut")
    ext_json = digiext.extract_digital([texts_cut])
    return ext_json,texts_cut
"""
if DATA:
    ext_json, texts_cut = data_prepare_word("./word_cut.txt", train_data)
    ext_json_eval, texts_cut_eval = data_prepare_word("./word_cut_eval.txt", eval_data)
    log("> ext_json")
    log(ext_json[0]['wx'])
    wx_word = ext_json[0]['wx']
    wx_word = re.sub(' ','',wx_word)
    wx_word = wx_word.split(',')

    log("wx_word")
    log(wx_word)

    log("获得规则提取结果 wx ")
"""
def marker_eve_char(f, wx, word):
    _l = len(word)
    if _l == 1:
        if word == "\n":
            f.write("\n")
        elif word == "\r":
            f.write("\r")
    elif _l == 2:
        if wx == False:
            f.write("%s/o "%word[0])
            f.write("%s/o "%word[1])
        else:
            word = replaceNC(word)
            f.write("%s/w "%word[0])
            f.write("%s/n "%word[1])
    else:
        if wx == False:
            ct = 0
            f.write("%s/o "%word[ct])
            while(1):
                ct+=1
                if _l>ct+1:
                    f.write("%s/o "%word[ct])
                    continue
                break
            f.write("%s/o "%word[ct])
        else:
            word = replaceNC(word)
            ct = 0
            f.write("%s/w "%word[ct])
            while(1):
                ct+=1
                if _l>ct+1:
                    f.write("%s/i "%word[ct])
                    continue
                break
            f.write("%s/n "%word[ct])


def data_prepare(filename, texts):
    with open(filename, "a+") as f:
         for word in jieba.cut(texts,HMM=True):
             if word in wx_word:
                 log("找到wx_word")
                 log(word)
                 marker_eve_char(f, True, word)
             else:
                 marker_eve_char(f, False, word)


def data_prepare_base(filename, cut_data_gen):
    cut_data = list()
    cut_data = cut_data_gen
    cnt = 0
    with open(filename, "a+") as f:
      for i in cut_data:
         print("> the sentence :", i)
         i = trans_char_num(i)
         dc_ = dex.handle_one_sent(rd.regDict, i)
         print("> the dc_:", dc_)
         for j in jieba.cut(i,HMM=True):
           #if len(re.findall("[a-zA-Z_0-9]",j))<1:
           #    continue
           #print(rd)
           #print(rd.regDict)
           if dc_ == "":
               dc_ = dict()
               dc_['wx'] = ""
           
           if len(dc_['wx'])>0:
               print(dc_['wx'])
           _l = len(j)
           if _l==1:
             if j in dc_['wx'].split(','):
               f.write("%s/w "%j)
             else:
               if j == "\n":
                   f.write("\n")
               elif j == "\r":
                   f.write("\r")
               #elif j == "3":
               #    f.write("3/s")
               #elif j == "C":
               #    f.write("C/s")
               else:
                   f.write("%s/s "%j)
           if _l==2:
             if j in dc_['wx'].split(','):
               f.write("%s/w "%j[0])
               f.write("%s/w "%j[1])
             else:
               #if j[0] == "3":
               #    f.write("3/s")
               #elif j[0] == "C":
               #    f.write("C/s")
               #else:
               #    f.write("%s/b "%j[0])
               f.write("%s/b "%j[0])
               f.write("%s/e "%j[1])
           if _l>2:
               if j in dc_['wx'].split(','):
                 print("> amazing j is in dc_['wx']")
                 ct = 0
                 f.write("%s/w "%j[ct])
                 while(1):
                   ct+=1
                   if _l>ct+1:
                       f.write("%s/w "%j[ct])
                       continue
                   break
                 f.write("%s/w "%j[ct])
               else:
                 ct = 0
                 f.write("%s/b "%j[ct])
                 while(1):
                   ct+=1
                   if _l>ct+1:
                       f.write("%s/m "%j[ct])
                       continue
                   break
                 f.write("%s/e "%j[ct])

def isNumChar(sent):
    s = re.findall("[a-zA-Z_0-9]", sent)
    if len(s)>0:
        return True
    return False

def isWx(sent):
    s = re.findall("微信", sent)
    if len(s)>0:
        return True
    return False

def read_txt(pathname):
    with open(pathname) as f:
        if DEBUG:
          #cont = list()
          #for i in range(100):
          #    cont.append(f.readline())
          #return cont
          #cont=f.readlines()[:1000]
          cont = ""
          for i in range(500000):
              if i%10000 ==1:
                  print("> read_txt line", i)
              _this = f.readline()
              if isWx(_this):
                  cont+=_this
                  cont+="\n"
          print("> read_txt finish")
          return cont

def read_shandong_addr_txt(pathname):
    with open(pathname) as f:
        if DEBUG:
          #cont = list()
          #for i in range(100):
          #    cont.append(f.readline())
          #return cont
          cont = ""
          for i in range(100):
              cont+=f.readline()
          return cont

# ## 1. 数据预处理

# In[2]:

# 以字符串的形式读入所有数据
###with open('../data/clus_text.txt', 'r') as inp:
#with open('raw_data/msr_train.txt', 'rb') as inp:

#with open('beijing110_cont.txt', 'r') as inp:
    #texts = inp.read().decode('gbk')
#    texts = inp.read()

#print(type(texts))

#sentences = texts.split("\n")  # 根据换行切分

import jieba
"""
print(">  读取 text 文件")
if DEBUG:
    texts = read_txt("/home/siyuan/data/beijing110_cp.txt")
else:
    sentences = read_shandong_addr_txt("/home/siyuan/bond_risk/bond_risk/shandong_add3.txt")
"""


def clear_marker(s):
    return re.sub("[\,\.\;'\]\[\]；\（\）\(\)，：．！？＂＂＇＇]","",s)

#texts = clear_marker(texts)

"""""
with open("/home/siyuan/bond_risk/bond_risk_sec/tagtext_cut.txt"):
    print(texts)
    texts_cut = f.readline()
    #sentences = texts.split("\n")
    sentences = texts_cut

with open("/home/siyuan/bond_risk/bond_risk_sec/tagtext_cut_eval.txt")
    print(texts)
    texts_cut_eval = f.readline()
    #sentences = texts.split("\n")
    sentences_eval = texts_cut_eval
"""
#print(sentences)
# 将不规范的内容（如每行的开头）去掉
def clean(s): 
    print("> clean before", s)
    s = re.sub("“",'"',s)
    s = re.sub("”",'"',s)
    s = re.sub("‘","'",s)
    s = re.sub("’","'",s)
    s = re.sub("，",",",s)
    s = re.sub("。",".",s)
    s = re.sub("：",":",s)
    s = re.sub("！","!",s)
    s = re.sub("？","?",s)
    s = re.sub("（","(",s)
    s = re.sub("）",")",s)
    s = re.sub("：",":",s)
    s = re.sub("；",";",s)
    s = re.sub("，",",",s)
    s = re.sub("[ ]+"," ",s)
    s = re.sub("\/","",s)
    print("> after", s)
    #s = re.sub(" ",",",s)

    if not DEBUG:
        if '“/s' not in s:  # 句子中间的引号不应去掉
            return s.replace(' ”/s', '')
        elif '”/s' not in s:
            return s.replace('“/s ', '')
        elif '‘/s' not in s:
            return s.replace(' ’/s', '')
        elif '’/s' not in s:
            return s.replace('‘/s ', '')
        else:
            return s
    return s
"""
print("> 拼接词语", len(sentences))
texts = ''.join(map(clear_marker, sentences)) # 把所有的词拼接起来
texts_eval = ''.join(map(clear_marker, sentences_eval)) # 把所有的词拼接起来
print('Length of texts is %d' % len(texts))
print('Example of texts: \n', texts[:3])

data_prepare("tagtext_cut.txt", texts)
data_prepare("tagtext_cut_eval.txt", texts_eval)
log("将切分后的词语保存如 tagtext_cut.txt")
log(texts)
#data_prepare("tagtext.txt", texts.split('\n'))
log("tagtext_cut.txt, 写入完毕，开始运算")
"""
def replace_keyword(wx_let_reg_exp, sent):
    for i in wx_let_reg_exp:
        print("> replace_keyword: ", i)
        fil_ = re.findall(i, sent)
        if len(fil_)==1:
            print("> replace_keyword: ", fil_)
            print("> replace_keyword: there is a wx kw")
            sent[2] = "w"
        pass
    print("> replace_keyword")

def load_json(filename):
    with open(filename, "r") as f:
        dc = json.loads(f.read())
        return dc

def save_json(dict_, filename):
    with open(filename, "w+") as f:
        f.write(json.dumps(dict_))
        return 0

'''''
wx_let = list()

ext_result = load_json("/home/siyuan/svn/algor/src/iba/dmp/gongan/storm_crim_classify/extcode/ext_dic.json")
for i in ext_result.keys():
    if ext_result[i] == "":
        continue
    _wx = ext_result[i]["wx"]
    if len(_wx)>0:
        _l = _wx.split(",")
        wx_let.extend(_l)
        wx_let = list(set(wx_let))

# all the exp for filter the content
wx_let_reg_exp = list()
for i in wx_let:
    temp_ = ""
    for j in i:
       temp_+="%s\/%s "%(j,".") 
    wx_let_reg_exp.append(temp_)

with open("tagtext_rep.txt","w+") as f:
  with open("tagtext.txt","r") as g:
    sentences = g.readlines()
    for sent in sentences:
        basesent = re.sub("/[bems]", "", sent)
        basesent = re.sub(" ", "", basesent)
        print("> basesent", basesent)
        wx_ = dex.handle_one_sent(rd.regDict, basesent)
        print("> sent write ", sent)
        print("> wx_: ", wx_)
        print("> wx_: ", type(wx_))
        #sent = replace_keyword(wx_, sent)
        if sent == None:
            continue
        f.write(sent)
        f.write("\n")

'''
#sentences = sentences.split("\n")
#with open("./tagtext_cut.txt", "r") as f:
#    sentences = f.read()
#print(sentences)

if True: #False:
  with open("wxLst.json", "r") as w:
    wx_word = json.loads(w.read())

  print("type wx_word: ", type(wx_word))

  with open("train", "r") as f:
    for line in f.readlines():
        data_prepare("train_cut.txt", line)

  with open("eval", "r") as g:
    for line in g.readlines():
        data_prepare("train_cut_evalu.txt", line)
else:
    pass
    
def clean_base(s): 
    if '“/s' not in s:  # 句子中间的引号不应去掉
        return s.replace(' ”/s', '')
    elif '”/s' not in s:
        return s.replace('“/s ', '')
    elif '‘/s' not in s:
        return s.replace(' ’/s', '')
    elif '’/s' not in s:
        return s.replace('‘/s ', '')
    else:
        return s

# 重新以换行符来划分
with open("train_cut.txt","r") as f:
    texts = f.read()
#sentences = re.split('[，。！？、‘’“”]/[bems]', texts)
sentences = re.split('[\r\n]', texts)


with open("train_cut_evalu.txt","r") as f:
    texts_eval = f.read()
#sentences = re.split('[，。！？、‘’“”]/[bems]', texts)
sentences_eval = re.split('[\r\n]', texts_eval)

print('Sentences number:', len(sentences))
log('Sentences number:')
log(len(sentences))
print('Sentence Example:\n', sentences[:3])
log('Sentence Example')
log(sentences[:3])

# In[4]:

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

datas = list()
labels = list()
datas_eval = list()
labels_eval = list()
print('Start creating words and tags data ...')
for sentence in tqdm(iter(sentences)):
        result = get_Xy(sentence)
        if result:
            datas.append(result[0])
            labels.append(result[1])

print('Start creating words and tags data ...')

for sentence in tqdm(iter(sentences_eval)):
        result = get_Xy(sentence)
        if result:
            datas_eval.append(result[0])
            labels_eval.append(result[1])
        else:
            datas_eval.append("\n")
            labels_eval.append("\n")


print('Length of datas is %d' % len(datas)) 
print('Example of datas: ', datas[0])
print('Example of labels:', labels[0])


df_data = pd.DataFrame({'words': datas, 'tags': labels}, index=list(range(len(datas))))
#　句子长度
df_data['sentence_len'] = df_data['words'].apply(lambda words: len(words))
df_data.head(2)

# 句子长度的分布
import matplotlib.pyplot as plt
df_data['sentence_len'].hist(bins=100)
plt.xlim(0, 100)
plt.xlabel('sentence_length')
plt.ylabel('sentence_num')
plt.title('Distribution of the Length of Sentence')
plt.show()


# 从上图可以看到，在使用标点进行分割后，绝大部分的句子长度小于30个字。因为一般情况下，我们训练网络的时候都喜欢把输入 padding 到固定的长度，这样子计算更快。因此我们取 32 作为句子长度，超过 32 个字的将把多余的字去掉，少于 32 个字的将用特殊字符填充。

# 1.用 chain(*lists) 函数把多个list拼接起来
from itertools import chain
all_words = list(chain(*df_data['words'].values))

# 2.统计所有 word
sr_allwords = pd.Series(all_words)
sr_allwords = sr_allwords.value_counts()
set_words = sr_allwords.index
set_ids = list(range(1, len(set_words)+1)) # 注意从1开始，因为我们准备把0作为填充值
tags = ['w','i','n','o']
tag_ids = list(range(len(tags)))
print("> tag_ids")
print(tag_ids)

# 3. 构建 words 和 tags 都转为数值 id 的映射（使用 Series 比 dict 更加方便）
word2id = pd.Series(set_ids, index=set_words)
id2word = pd.Series(set_words, index=set_ids)
tag2id = pd.Series(tag_ids, index=tags)
id2tag = pd.Series(tags, index=tag_ids)

vocab_size = len(set_words)
print('vocab_size={}'.format(vocab_size))

# 把 words 和 tags 都转为数值 id


max_len = 100

def X_padding(words):
    """把 words 转为 id 形式，并自动补全位 max_len 长度。"""
    ids = list(word2id[words])
    if len(ids) >= max_len:  # 长则弃掉
        return ids[:max_len]
    ids.extend([0]*(max_len-len(ids))) # 短则补全
    return ids

def y_padding(tags):
    """把 tags 转为 id 形式， 并自动补全位 max_len 长度。"""
    ids = list(tag2id[tags])
    if len(ids) >= max_len:  # 长则弃掉
        return ids[:max_len]
    ids.extend([0]*(max_len-len(ids))) # 短则补全
    return ids

#get_ipython().magic("time df_data['X'] = df_data['words'].apply(X_padding)")
#get_ipython().magic("time df_data['y'] = df_data['tags'].apply(y_padding)")

df_data['X'] = df_data['words'].apply(X_padding)
df_data['y'] = df_data['tags'].apply(y_padding)

# In[10]:


# 最后得到了所有的数据
X = np.asarray(list(df_data['X'].values))
y = np.asarray(list(df_data['y'].values))
print('X.shape={}, y.shape={}'.format(X.shape, y.shape))
print('Example of words: ', df_data['words'].values[0])
print('Example of X: ', X[0])
print('Example of tags: ', df_data['tags'].values[0])
print('Example of y: ', y[0])


# In[11]:


# 保存数据
import pickle
import os

if not os.path.exists('data/'):
    os.makedirs('data/')

"""
with open('data/data.pkl', 'wb') as outp:
    pickle.dump(X, outp)
    pickle.dump(y, outp)
    #get_ipython().magic('time pickle.dump(X, outp)')
    #get_ipython().magic('time pickle.dump(y, outp)')
    pickle.dump(word2id, outp)
    pickle.dump(id2word, outp)
    pickle.dump(tag2id, outp)
    pickle.dump(id2tag, outp)
print('** Finished saving the data.')    


# In[12]:


# 导入数据
import pickle
with open('data/data.pkl', 'rb') as inp:
    X = pickle.load(inp)
    y = pickle.load(inp)
    #get_ipython().magic('time X = pickle.load(inp)')
    #get_ipython().magic('time y = pickle.load(inp)')
    word2id = pickle.load(inp)
    id2word = pickle.load(inp)
    tag2id = pickle.load(inp)
    id2tag = pickle.load(inp)
"""
# 划分测试集/训练集/验证集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train,  test_size=0.2, random_state=42)
print('X_train.shape={}, y_train.shape={}; \nX_valid.shape={}, y_valid.shape={};\nX_test.shape={}, y_test.shape={}'.format(
    X_train.shape, y_train.shape, X_valid.shape, y_valid.shape, X_test.shape, y_test.shape))


# ## 2. 构造一个生成batch数据的类

# In[13]:


# ** 3.build the data generator
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
print("> data_trian.__next__()")
print(data_train.y.shape)
print(data_train.X.shape)
data_valid = BatchGenerator(X_valid, y_valid, shuffle=False)
data_test = BatchGenerator(X_test, y_test, shuffle=False)
print('Finished creating the data generator.')


# ## 3. Bi-directional lstm 模型

# ### 3.1 模型构造

# In[14]:


import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
from tensorflow.contrib import rnn
import numpy as np

'''
For Chinese word segmentation.
'''
# ##################### config ######################
decay = 0.85
max_epoch = 10
max_max_epoch = 60
timestep_size = max_len = 100           # 句子长度
vocab_size = 6500    # 样本中不同字的个数+1(padding 0)，根据处理数据的时候得到
input_size = embedding_size = 64       # 字向量长度
class_num = 7
hidden_size = 128    # 隐含层节点数
layer_num = 2        # bi-lstm 层数
max_grad_norm = 5.0  # 最大梯度（超过此值的梯度将被裁剪）

lr = tf.placeholder(tf.float32, [])
keep_prob = tf.placeholder(tf.float32, [])
batch_size = tf.placeholder(tf.int32, [])  # 注意类型必须为 tf.int32
model_save_path = 'ckpt/bi-lstm.ckpt'  # 模型保存位置


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
    log("embedding")
    log(embedding)
    log(X_inputs)
    log("X_inputs")
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

# In[16]:


def test_epoch(dataset):
    """Testing or valid."""
    _batch_size = 100
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
    print("batch_num")
    print(batch_num)
    mean_acc= _accs / batch_num     
    mean_cost = _costs / batch_num
    return mean_acc, mean_cost


sess.run(tf.global_variables_initializer())
tr_batch_size = 64 
#max_max_epoch = 6
display_num = 2  # 每个 epoch 显示是个结果
print("> data_train", data_train)
log("data_train.y.shape tr_batch_size")
log(data_train.y.shape)
log(tr_batch_size)
tr_batch_num = int(data_train.y.shape[0] / tr_batch_size)  # 每个 epoch 中包含的 batch 数
print("> tr_batch_num", tr_batch_num)

display_batch = int(tr_batch_num / display_num)  # 每训练 display_batch 之后输出一次
print("display_batch", display_batch)
saver = tf.train.Saver(max_to_keep=10)  # 最多保存的模型数量
for epoch in range(max_max_epoch):
    _lr = 1e-4
    if epoch > max_epoch:
        _lr = _lr * ((decay) ** (epoch - max_epoch))
    print('EPOCH %d， lr=%g' % (epoch+1, _lr))
    start_time = time.time()
    _costs = 0.0
    _accs = 0.0
    show_accs = 0.0
    show_costs = 0.0
    print("y_inputs:y_batch")
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
        print(display_batch)
        print(batch)
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


# 模型测试，现在给定一个字符串，首先应该把处理成正确的模型输入形式。即每次输入一个片段，（这里限制了每个片段的长度不超过 max_len=32）。每个字处理为对应的 id， 每个片段都会 padding 处理到固定的长度。也就是说，输入的是一个list， list 的每个元素是一个包含多个 id 的list。<br/>
# 即 [[id0, id1, ..., id31], [id0, id1, ..., id31], [], ...]

# In[17]:


# ** 导入模型
saver = tf.train.Saver()
best_model_path = 'ckpt/bi-lstm.ckpt-6'
saver.restore(sess, best_model_path)
#get_ipython().magic('time saver.restore(sess, best_model_path)')


def load_model():
    saver = tf.train.Saver()
    best_model_path = 'ckpt/bi-lstm.ckpt-6'
    saver.restore(sess, best_model_path)

# In[18]:


# 再看看模型的输入数据形式, 我们要进行分词，首先就要把句子转为这样的形式
X_tt, y_tt = data_train.next_batch(2)
print('X_tt.shape=', X_tt.shape, 'y_tt.shape=', y_tt.shape)
print('X_tt = ', X_tt)
print('y_tt = ', y_tt)


# ## 维特比解码
# 下面使用维特比的方法来计算的最优的状态序列。具体原理可以参考 隐马尔可夫模型。在 HMM 监督学习中，我们统计样本频数来得到转移概率和 

# In[19]:


# 利用 labels（即状态序列）来统计转移概率
# 因为状态数比较少，这里用 dict={'I_tI_{t+1}'：p} 来实现
# A统计状态转移的频数

def zy_mat():
    A = {
      'wi':1e-9,
      'wn':1e-9,
      'wo':1e-9,
      'iw':1e-9,
      'in':1e-9,
      'io':1e-9,
      'nw':1e-9,
      'ni':1e-9,
      'no':1e-9,
      'ow':1e-9,
      'oi':1e-9,
      'on':1e-9,
     }

    # zy 表示转移概率矩阵
    zy = dict()
    #for i in A.keys():
    #    A[i] = 0.1
    for label in labels:
        for t in range(len(label) - 1):
            key = label[t] + label[t+1]
            A[key] += 1.0
        
    log("生成A")
    log(A)

    zy['wi'] = A['wi'] / (A['wi'] + A['wo'] + A['wn'] + A['ww'])
    zy['wn'] = A['wn'] / (A['wi'] + A['wo'] + A['wn'] + A['ww'])
    zy['wo'] = A['wo'] / (A['wi'] + A['wo'] + A['wn'] + A['ww'])
    zy['ww'] = A['ww'] / (A['wi'] + A['wo'] + A['wn'] + A['ww'])

    zy['ii'] = A['io'] / (A['ii'] + A['iw'] + A['in'] + A['io'])
    zy['iw'] = A['io'] / (A['ii'] + A['iw'] + A['in'] + A['io'])
    zy['in'] = A['io'] / (A['ii'] + A['iw'] + A['in'] + A['io'])
    zy['io'] = A['io'] / (A['ii'] + A['iw'] + A['in'] + A['io'])


    zy['nn'] = A['nn'] / (A['ni'] + A['nw'] + A['nn'] + A['no'])
    zy['ni'] = A['ni'] / (A['ni'] + A['nw'] + A['nn'] + A['no'])
    zy['nw'] = A['nw'] / (A['ni'] + A['nw'] + A['nn'] + A['no'])
    zy['no'] = A['no'] / (A['ni'] + A['nw'] + A['nn'] + A['no'])

    zy['oo'] = A['oo'] / (A['oi'] + A['ow'] + A['on'] + A['oo'])
    zy['oi'] = A['oi'] / (A['oi'] + A['ow'] + A['on'] + A['oo'])
    zy['ow'] = A['ow'] / (A['oi'] + A['ow'] + A['on'] + A['oo'])
    zy['on'] = A['on'] / (A['oi'] + A['ow'] + A['on'] + A['oo'])


    keys = sorted(zy.keys())
    print('the transition probability: ')
    for key in keys:
        print(key, zy[key])
    
    zy = {i:np.log(zy[i]) for i in list(zy.keys())}
    return zy
zy = zy_mat()

def viterbi(nodes):
    print("viterbi nodes list")
    log("viterbi nodes list")
    log(nodes)
    """
    维特比译码：除了第一层以外，每一层有4个节点。
    计算当前层（第一层不需要计算）四个节点的最短路径：
       对于本层的每一个节点，计算出路径来自上一层的各个节点的新的路径长度（概率）。保留最大值（最短路径）。
       上一层每个节点的路径保存在 paths 中。计算本层的时候，先用paths_ 暂存，然后把本层的最大路径保存到 paths 中。
       paths 采用字典的形式保存（路径：路径长度）。
       一直计算到最后一层，得到四条路径，将长度最短（概率值最大的路径返回）
    """
    #paths = {'b': nodes[0]['b']} # 第一层，只有一个节点
    paths = {'w': nodes[0]['w'], 'o':nodes[0]['o']} # 第一层，只有两个节点
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
                log("path_last[-1]")
                log(path_last[-1])
                log("node_now")
                log(node_now)
                log("跳转是")
                log(path_last[-1])
                log(node_now)
                if path_last[-1] + node_now in list(zy.keys()): # 若转移概率不为 0 
                    sub_paths[path_last + node_now] = paths_[path_last] + nodes[layer][node_now] + zy[path_last[-1] + node_now]
                    log("迭代子路径")
                    log(sub_paths)
                    print(sub_paths)
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
    print("viterbi返回的排列")
    print(sr_paths.index[-1])
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
    if text:
        log("待检测的text")
        log(text)
        text_len = len(text)
        X_batch = text2ids(text)  # 这里每个 batch 是一个样本
        fetches = [y_pred]
        feed_dict = {X_inputs:X_batch, lr:1.0, batch_size:1, keep_prob:1.0}
        log("用于检查的数据　feed_dict")
        log(feed_dict)
        _y_pred = sess.run(fetches, feed_dict)[0][:text_len]  # padding填充的部分直接丢弃
        log("_y_pred")
        log(_y_pred)
        #nodes = [dict(list(zip(['x', 'w','i','n','s','b','m','e'], each))) for each in _y_pred]
        nodes = [dict(list(zip(['w','i','n','o'], each))) for each in _y_pred]
        print("> 使用模型训练，找出节点:", nodes)
        log("nodes 模型训练出的节点")
        log(nodes)
        #pd.DataFrame([list(len(text)), list(text),])
        tags = viterbi(nodes)
        log("原文内容")
        log(text)
        log("输出概率")
        log(nodes)
        log("预测内容")
        log(tags)
        return tags
        """
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
                log(text[i])
                #log(words[-1])
                #words[-1] += text[i]
        log("打印 words 仅仅包括　微信名")
        log(words)
        return words
        """
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
    log("测试用的分词结果　result.extend(simple_cut(sentence[start:]))")
    result = simple_cut(sentence)
    log(result)
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
def data_init():
    #clstxt.prepare_train_eval()
    with open("~/gensim_word2vec/train", "r") as f:
        train = f.readlines()
    with open("~/gensim_word2vec/evalu", "r") as f:
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


"""""
if __name__ == "__main__"():

    data_init()
    data_load()
    modul_init()
    train()
    evaluate()
"""    



"""

# In[21]:


# 例一
#sentence = '人们思考问题往往不是从零开始的。就好像你现在阅读这篇文章一样，你对每个词的理解都会依赖于你前面看到的一些词，      而不是把你前面看的内容全部抛弃了，忘记了，再去理解这个单词。也就是说，人们的思维总是会有延续性的。'
sentence = '我的微信号码是wxid13678028750'
result = cut_word(sentence)
rss = ''
for each in result:
    rss = rss + each + ' / '
print(rss)


# In[22]:


# 例二
#sentence = '天舟一号是我国自主研制的首艘货运飞船，由于它只运货，不送人，所以被形象地称为太空“快递小哥”。    它采用两舱式结构，直径较小的是推进舱，直径较大的为货物舱。其最大直径达到3.35米，飞船全长10.6米，载荷能力达到了6.5吨，    满载货物时重13.5吨。如果此次满载的话，它很可能将成为中国发射进入太空的质量最大的有效载荷。甚至比天宫二号空间实验室还大，    后者全长10.4米，直径同为3.35米，质量为8.6吨。'
sentence = '我微信号:wxid13678028750'
result = cut_word(sentence)
rss = ''
for each in result:
    rss = rss + each + ' / '
print(rss)


# In[23]:


# 例三
#sentence = '南京市长江大桥'
sentence = '13678028750微信号'
result = cut_word(sentence)
rss = ''
for each in result:
    rss = rss + each + '/ '
print(rss)

cc=100
with open("beijing110_cont.txt", "r") as f:
    with open("beijing110_result.txt", "a+") as g:
        while(cc>0):
            cc-=1
            line = f.readline()
            if len(re.findall("[a-zA-Z_0-9]", line))>0:
                line = clear(line)
                cuts_ = cut_word(line)
                ch_ = ""
                for ch in cuts_:
                    ch_+=ch
                    g.write(ch_)
                    print(ch)
                    g.write("\n")

# 结论：本例子使用 Bi-directional LSTM 来完成了序列标注的问题。本例中展示的是一个分词任务，但是还有其他的序列标注问题都是可以通过这样一个架构来实现的，比如 POS（词性标注）、NER（命名实体识别）等。在本例中，最后的分词效果还不是非常好，但已经达到了实用的水平，而且模型也只是粗略地跑了一遍，还没有进行任何的参数优化。最后的维特比译码中转移概率根据训练语料进行统计。
# 
# 看到最后一个 "南京/ 市长江大桥/" 的结果时，心里不由的一阵哀伤...
# 
# 在模型构造中，我们对 Bi-directional LSTM 模型进行了比较详细的展开分析，从而对模型有了深入的理解。这很大程度上也得益于 TensorFlow 比较底层，如果是用 keras 框架的话，虽然只需要短短的几行代码就搞定了，但是我们对于模型的理解估计不会这么深入。

"""
