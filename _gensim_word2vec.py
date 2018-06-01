#encoding=utf-8
# gensim for word2 vec

import re
import os
from gensim.models import word2vec
from gensim.models import Word2Vec  
#from . import data_helper
import jieba
from gensim import corpora
import gensim
import traceback
import pandas as pd
import collections
import math
import os
import sys
import argparse
import random
from tempfile import gettempdir
import zipfile
import numpy as np


class wd2vec(object):
    def __init__(self):
        pass
        self.modelpath = "gensim_word2vec.model"
        self.txtfilepath = "/home/siyuan/data/alldata.txt"
        self.model = ""
        self.texts = ""
        self.sentences = list()
        self.corpus = ""
        self.tfidfmodel = ""
        self.load_txt()
        self.word2vec()
        self.texts_bind()
        self.gen_dictionary()
        self.tfidf()

    def texts_bind(self):
        for i in self.sentences:
            self.texts+=i

    def load_txt(self):
        with open(self.txtfilepath, "r") as f:
            lines = f.readlines()
            print((lines[:3]))
            self.sentences = lines[:100]
            #return lines

    def word2vec(self):
        self.model = Word2Vec(self.sentences, sg=1, size=100,  window=5,  min_count=3,  negative=3, sample=0.001, hs=1, workers=4)  
        self.model.save(self.modelpath)

    def load_model(self):
        self.model = Word2Vec.load(self.modelpath) 
        return self.model

    def most_similar(self, ch):
        return self.model.most_similar(ch)

    def similar(self, ch, ch_):
        return self.model.similarity(ch, ch_)

    def char2vec(self, ch):
        return self.model[ch]

    def gen_dictionary(self):
        dictionary = corpora.Dictionary(self.texts)
        corpus = [dictionary.doc2bow(text) for text in self.texts]
        # print corpus[0] # [(0, 1), (1, 1), (2, 1)]
        return corpus

    def tfidf(self):
        self.tfidfmodel = gensim.models.TfidfModel(self.corpus)

    def tfidf_sent(self, sent):
        return self.tfidf_sent[sent]

#md = wd2vec()
#a = md.similar("你","我")
#b = md.most_similar("你")
#c = md.char2vec("他")
#print(a,b,c)

def data_clear():
    _l = list()
    for sent in md.sentences:
        _d = jieba.cut(sent, HMM=True)
        for i in _d:
            _l.append(sub(i))
    return "".join(_l)
        
def sub(s): 
    s = re.sub("“",'',s)
    s = re.sub("”",'',s)
    s = re.sub("‘","",s)
    s = re.sub("’","",s)
    s = re.sub("，","",s)
    s = re.sub("。","",s)
    s = re.sub("：","",s)
    s = re.sub("！","",s)
    s = re.sub("？","",s)
    s = re.sub("（","",s)
    s = re.sub("）","",s)
    s = re.sub("：","",s)
    s = re.sub("，","",s)
    s = re.sub("[ ]+", "",s)
    s = re.sub("\(", "",s)
    s = re.sub("\)", "",s)
    s = re.sub("\[", "",s)
    s = re.sub("\]", "",s)
    return s

from gensim.models import Word2Vec  

sentences = []

with open("/home/siyuan/data/ner_sample.txt", "r") as f:
    cont = f.read()
    lines = cont.split("\n")
    idx = np.random.permutation(len(lines))
    for i in idx[:50000]: 
        sentences.append(list(jieba.cut(lines[i], HMM=True)))

print(sentences)

model = Word2Vec(sentences, sg=1, size=100,  window=5,  min_count=5,  negative=3, sample=0.001, hs=1, workers=4) 

model.save("./word2vec_gensim.model")
def isstopword(word):
    if word == "_":
        return True
    if len(re.findall("(.先生|.女士)",word))>0:
        return True
    else:
        return False


model_path = "./word2vec_gensim.model"
model= word2vec.Word2Vec.load("./word2vec_gensim.model")

y2=model.similarity("冒充", "公司")
print(y2)
#for i in model.most_similar("公司"):
#    print(i[0],i[1])

import collections

words_lst = []
target_words_lst = []


from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def spot_vec():
    model = word2vec.Word2Vec.load(model_path)  
    vocab = model.wv.vocab  
    word_vector = {}
    for word in vocab:    
        word_vector[word] = model[word]
    return word_vector

word_vector = spot_vec()
print("word_vector ok")

ndarr_lst = []

def get_arr(word):
    return word_vector[word]

def simi_between_num(sentences):
    for sentence in sentences:
        flag = False
        tempword = ""
        #dq = collections.deque(maxlen=3)
        #dq.append("_")
        #dq.append("_")
        #dq.append("_")
        for word in sentence:
            if flag:
                if isstopword(word):
                    continue
                try:
                   ndarr_lst.append(get_arr(word))
                   words_lst.append(word)
                   target_words_lst.append(tempword)
                   flag = False
                   break
                except:
                    continue
            #if isstopword(word):
            #    continue
            #dq.append(word)
            if len(re.findall(r"\d{11,}",word))>0:
                flag =True
                tempword = word
                    #y2=model.similarity(dq[0], dq[2])
                    #print("\n 目标词:", dq[1], "    前后相似度:", y2, "    ", dq[0], dq[2])

simi_between_num(sentences[:1000])
print("simi_between_num ok")
import numpy as np
ndarr = np.array(ndarr_lst).reshape(len(words_lst), 100)

def plot_with_labels(low_dim_embs, labels, filename):
    assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
    plt.figure(figsize=(38, 38))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        if label in warnning_word:
            pass
            print(label)
            plt.scatter(x, y, c='b')
        else:
            plt.scatter(x, y, c='r')
            pass
        plt.annotate(
            label,
            xy=(x, y),
            xytext=(random.randint(0,150), random.randint(0,150)),
            textcoords='offset points',
            ha='right',
            va='bottom')
    plt.savefig(filename)


warnning_word = ['报警','报','再报','报案']
def tsne(ndarr, target_words_lst):
    try:
        tsne = TSNE(perplexity=100, n_components=2, init='pca', n_iter=1000, method='exact')
        low_dim_embs = tsne.fit_transform(ndarr)
        plot_with_labels(low_dim_embs, target_words_lst, os.path.join(gettempdir(), 'tsne.png'))
        df = pd.DataFrame(columns=['words','targetwords','x','y'])
        df['words'] = words_lst
        df['targetwords'] = target_words_lst
        df['x'] = low_dim_embs[:,0]
        df['y'] = low_dim_embs[:,1]
        df.to_csv("_df_word_vec.csv")
        print("the pic save ok")
        return df.reset_index()
    except ImportError as ex:
        print('Please install sklearn, matplotlib, and scipy to show embeddings.')
        print(ex)

df = tsne(ndarr,target_words_lst) 
print("plot ok !")
print(df.index)
print(df.columns)

with open("key_target.txt",'w+') as f:
    for i in range(len(target_words_lst)):
        print(words_lst[i],"    ", target_words_lst[i])
        f.write("\n"+words_lst[i]+"    "+target_words_lst[i])

y2=model.similarity("报警", "报")
print(y2)
y2=model.similarity("报警", "再报")
print(y2)
y2=model.similarity("报", "再报")
print(y2)

import tensorflow as tf
import numpy as np

#构造一些离散的点
#x_data = df['x']
#y_data = df['y']
def number_to_flag(n):
    if n in warnning_word:
        return 1
    return 0

df['mark'] = df['words'].map(number_to_flag)

print("\n> df")
print(df.iloc[:10,:])

dpos = df[df['mark']==1]
dneg = df[df['mark']==0]


rnd = 10
def data_generate(df):
    _p = []
    pos_neg= random.randint(0,1)
    if pos_neg==0:
      for i in range(rnd):
        _l = []
        _l.append(dpos.iloc[np.random.randint(0,len(dpos)-1), 3])
        _l.append(dpos.iloc[np.random.randint(0,len(dpos)-1), 4])
        _p.append(_l)
    else:
      for i in range(rnd):
        _l = []
        _l.append(dneg.iloc[np.random.randint(0,len(dneg)-1), 3])
        _l.append(dneg.iloc[np.random.randint(0,len(dneg)-1), 4])
        _p.append(_l)
    _p = np.array(_p).reshape(rnd,2)
    #print("\n===\n>", _p, _p.shape)
    return _p

#tensorflow建模
x= tf.placeholder("float", [None, 1])
y_input = tf.placeholder("float", [None, 1])

y = tf.placeholder("float", [None, 1])

Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))  #设置权重的初始值1以及变化范围（-1， 1）
Weights2 = tf.Variable(tf.random_uniform([1], -1.0, 1.0))  #设置权重的初始值1以及变化范围（-1， 1）
biases = tf.Variable(tf.zeros([1])) #设置偏移量的初始值为0
biases2 = tf.Variable(tf.zeros([1])) #设置偏移量的初始值为0
_y = Weights * x+ biases #对以上离散的点建立一个一次函数的数学模型
y = Weights2 * _y + biases2 #对以上离散的点建立一个一次函数的数学模型
optimizer = tf.train.GradientDescentOptimizer(1e-4)
init = tf.global_variables_initializer()
loss = tf.reduce_mean(tf.square(y-y_input))
train = optimizer.minimize(loss)
sess = tf.Session()
sess.run(init)

#训练
"""""
for step in range(100000):
    try:
        _p = data_generate(df)
        sess.run(train, feed_dict={x:np.array(_p[:,0]).reshape(rnd,1), y_input:np.array(_p[:,1]).reshape(rnd,1)})
        if step % 2000 == 0:
            _y_, _loss = sess.run([y,loss], feed_dict={x:np.array(_p[:,0]).reshape(rnd,1), y_input:np.array(_p[:,1]).reshape(rnd,1)})
            print("\n>", step,  _loss)
            #print("\n>", _p)
    except:
        traceback.print_exc()
"""
sess.close()


