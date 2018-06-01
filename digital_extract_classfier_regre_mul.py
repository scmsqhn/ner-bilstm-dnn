
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

tags = ['z','b','i','e','s','p','h','n','u','v','x','d','t','f','Q','q','k','c','r','R']
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

envs = dict()

#============================================================================================================
#
# block1: read the txt from file, sav to conti and lines
#
#============================================================================================================
f = open("./cut_char_without_marker.txt","r")
envs['f'] = f
envs['cont'] = f.read()
envs['sentences'] = envs['cont'].split("\n")
#============================================================================================================
#
# end
#
#============================================================================================================

def log(n):
    with open("./log.txt", "a+") as f:
        f.write("\n> "+str(n))
        f.write("\n")

def replaceNC(line):
    #line = re.sub("[0-9]","3",line)
    #line = re.sub("[A-Z_a-z]","C",line)
    return line

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
    log("\n> ext_json")
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
            f.write("%s/o "%word[0])
            f.write("%s/o "%word[1])
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
            f.write("%s/b "%word[ct])
            while(1):
                ct+=1
                if _l>ct+1:
                    f.write("%s/i "%word[ct])
                    continue
                break
            f.write("%s/i "%word[ct])


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
         print("\n> the sentence :", i)
         i = trans_char_num(i)
         dc_ = dex.handle_one_sent(rd.regDict, i)
         print("\n> the dc_:", dc_)
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
                 print("\n> amazing j is in dc_['wx']")
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
                  print("\n> read_txt line", i)
              _this = f.readline()
              if isWx(_this):
                  cont+=_this
                  cont+="\n"
          print("\n> read_txt finish")
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

def clear_marker(s):
    return re.sub("[\,\.\;'\]\[\]；\（\）\(\)，：．！？＂＂＇＇]","",s)

def replace_keyword(wx_let_reg_exp, sent):
    for i in wx_let_reg_exp:
        print("\n> replace_keyword: ", i)
        fil_ = re.findall(i, sent)
        if len(fil_)==1:
            print("\n> replace_keyword: ", fil_)
            print("\n> replace_keyword: there is a wx kw")
            sent[2] = "w"
        pass
    print("\n> replace_keyword")

def load_json(filename):
    with open(filename, "r") as f:
        dc = json.loads(f.read())
        return dc

def save_json(dict_, filename):
    with open(filename, "w+") as f:
        f.write(json.dumps(dict_))
        return 0

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

#============================================================================================================
#
# block2: from lines generate the data and tags
#         and then seperate into train and eval
#
#============================================================================================================

envs['datas'] = list()
envs['labels'] = list()
envs['datas_eval'] = list()
envs['labels_eval'] = list()

print('Start creating words and tags data ...')

#envs['words'],envs['tags'] = get_Xy(envs['lines'])
for sentence in tqdm(iter(envs['sentences'])):
        result = get_Xy(sentence)
        if result:
            envs['datas'].append(result[0])
            envs['labels'].append(result[1])
        #else:
        #    datas_eval.append("\n")
        #    labels_eval.append("\n")

print("\nwords sample: ", envs['datas'][:12])
print("\nwords sample: ", envs['labels'][:12])
print("\nsentencessample: ", envs['sentences'][:1])
#============================================================================================================
#
# end: output datas labels sentences
#
#============================================================================================================

#============================================================================================================
#
# block3: calcu the sentence len, generate the dictionary 
#
#============================================================================================================
datas = envs['datas']
labels = envs['labels']
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

# 1.concat all list with chain(*lists)
from itertools import chain
all_words = list(chain(*df_data['words'].values))

# 2.统计所有 word
sr_allwords = pd.Series(all_words)
sr_allwords = sr_allwords.value_counts()
set_words = sr_allwords.index
set_ids = list(range(1, len(set_words)+1)) # 注意从1开始，因为我们准备把0作为填充值
print("\ntags: ", tags)
tag_ids = list(range(len(tags)))
print("\n> tag_ids")
print(tag_ids)

# 3. use series instead of dict to handle the kv
word2id = pd.Series(set_ids, index=set_words)
id2word = pd.Series(set_words, index=set_ids)
tag2id = pd.Series(tag_ids, index=tags)
id2tag = pd.Series(tags, index=tag_ids)

vocab_size = len(set_words)
print('vocab_size={}'.format(vocab_size))

#============================================================================================================
#
# end
#
#============================================================================================================

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

#============================================================================================================
#
# block 4: formula to 100 * 1 char sentence, get all the data totally be handled
#
#============================================================================================================

max_len = 100
df_data['X'] = df_data['words'].apply(X_padding)
df_data['y'] = df_data['tags'].apply(y_padding)

X = np.asarray(list(df_data['X'].values))
y = np.asarray(list(df_data['y'].values))
#encs['df_data'] = df_data

#============================================================================================================
#
# end
#
#============================================================================================================

# 最后得到了所有的数据
print('X.shape={}, y.shape={}'.format(X.shape, y.shape))
print('Example of words: ', df_data['words'].values[0])
print('Example of X: ', X[0])
print('Example of tags: ', df_data['tags'].values[0])
print('Example of y: ', y[0])

import pickle
import os

if not os.path.exists('/home/siyuan/data/'):
    os.makedirs('/home/siyuan/data/')

with open('/home/siyuan/data/data.pkl', 'wb') as outp:
    pickle.dump(X, outp)
    pickle.dump(y, outp)
    #get_ipython().magic('time pickle.dump(X, outp)')
    #get_ipython().magic('time pickle.dump(y, outp)')
    pickle.dump(word2id, outp)
    pickle.dump(id2word, outp)
    pickle.dump(tag2id, outp)
    pickle.dump(id2tag, outp)
print('** Finished saving the data.')    

"""
for i in envs:
    print(i)
with open("envs.json", "w") as f:
    f.write(json.dumps(envs))
"""
