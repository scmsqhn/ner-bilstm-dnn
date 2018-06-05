import sys
#sys.path.append("/home/siyuan/svn/algor/src/iba/dmp/gongan/storm_crim_classify/extcode")
#sys.path.append("/home/siyuan/svn/algor/src/iba/dmp/gongan/storm_crim_classify/extcode")
#sys.path.append("/home/distdev/anaconda3/envs/qq/lib/python3.6/site-packages")

import traceback
#import digital_info_extract as dex
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
#from tqdm import tqdm
import time
import os
import jieba
import collections
#sys.path.append("/home/siyuan/gensim_word2vec")
#import classifier_text as clstxt
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sklearn.utils
from sklearn.utils import shuffle
#import data_generator
sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")
import myconfig as config
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import json
import pickle
from sklearn.model_selection import train_test_split
import myconfig as config

max_len = 100
"""

with open('/home/police/iba/dmp/gongan/storm_crim_classify/extcode/data3.pkl', 'rb') as inp:
    X = pickle.load(inp)
    y = pickle.load(inp)
    word2id = pickle.load(inp)
    id2word = pickle.load(inp)
    tag2id = pickle.load(inp)
    id2tag = pickle.load(inp)
with open('/home/police/iba/dmp/gongan/storm_crim_classify/extcode/data3.pkl', 'rb') as inp:
    X = pickle.load(inp)
    y = pickle.load(inp)
    word2id = pickle.load(inp)
    id2word = pickle.load(inp)
    tag2id = pickle.load(inp)
    id2tag = pickle.load(inp)
"""
def su(l):
    return re.sub(r"[{}' \n\t]","",l)

tags = config.tags
tfconfig = tf.ConfigProto()
tfconfig.gpu_options.allow_growth = True
sess = tf.Session(config=tfconfig)

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

sess.run(tf.global_variables_initializer())

new_saver=tf.train.import_meta_graph(os.path.join(config.CUR_PATH,'data/auto_encode.ckpt-4.meta'))
new_saver.restore(sess,os.path.join(config.CUR_PATH,'data/auto_encode.ckpt-4'))
graph = tf.get_default_graph()

x=tf.get_collection("x")[0]
scale=tf.get_collection("scale")[0]
cost=tf.get_collection("cost")[0]
reconstruction=tf.get_collection("reconstruction")[0]

scale = 1

import word2vec
from word2vec._gensim_word2vec import wd2vec

def gen_w2v():
   w2v = wd2vec()
   return w2v

n_samples = 200000
training_epochs = 100
batch_size = 32
n_input = 100
display_step = 1

df = pd.read_csv(os.path.join(config.CUR_PATH, "data/_df_word_vec.csv"))
def data_generate(w2v, df=df):
   word2vec, model = w2v.spot_vec()
   lines = w2v.load_txt(w2v.txtfilepath, False)
   _random_lst = np.random.permutation(len(lines))
   _arrlst = []
   cnt = 0
   for _id in _random_lst:
      words = lines[_id].split(" ")
      for word in words:
         try:
             _word_arr = word2vec[word] # renturn one (100,) shape array
             _arrlst.append(_word_arr)
         except KeyError:
             #print("\n> the word", word, " is not in the vocab")
             continue
         cnt+=1
         if cnt % batch_size == 0:
             _arr = np.array(_arrlst)
             _arrlst = []
             cnt = 0
             yield _arr
   print("\n> all data read finish")

def _dis(vec1, vec2):
  dist = numpy.linalg.norm(vec1 - vec2) 
  return dist

#from word2vec._gensim_word2vec import wd2vec
f = open("./pred_auto_encode.txt", "w+")
w2v = gen_w2v()
x_test = data_generate(gen_w2v())
word2vec, model = w2v.spot_vec()


while(1):
  message = input("\n> How many sentence u wanna to get, if input '0' I'll quit ?\n")
  print("\n> sentence is ", message, 'pls be patients...')
  _lst = []
  print("\n 主题抽取:")
  for i in range(int(message)//1):
    _lnext = x_test.__next__()
    index = []
    print("\n 原输入为:")
    for i in _lnext:
      index.append(model.similar_by_vector(i)[0][0])
    
    _feed_dict_x = np.array(_lnext).reshape(32, 100)
    _y_pred = sess.run([reconstruction], feed_dict={x:_feed_dict_x})  # padding填充的部分直接丢弃

    print("\n 提取后的　词汇向量　转词")
    print('\n>_y_pred[0].shape: ',_y_pred[0].shape)
    columns = []

    for i in _y_pred[0]:
      columns.append(model.similar_by_vector(i)[-1][0])

    atten_df = pd.DataFrame(index=index, columns=columns)
    for i in index:
      for j in index:
        atten_df.iloc[i,j] = model.similarity(i, j)

    height = 32
    width= 32
    arr = np.zeros((height, width))
    for i in range(len(x)):
        arr[y[i], x[i]] = v[i]

    plt.matshow(arr, cmap='hot')
    plt.colorbar()
    plt.show()
    
    

def compare_predict(tags, predict, text):
    return pd.DataFrame([tags,predict, text])

def text2ids(text):
    """把字片段text转为 ids."""
    words = list(text)
    print(words)
    for i in words:
       print(i)
       j = word2id[i]
       print(j)
    #ids = []
    ids = list(word2id[words].values)
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
            print(_y_pred.shape)
            print(_y_pred[0])
            print(_y_pred[-1])
            #nodes = [dict(list(zip(['x', 'w','i','n','s','b','m','e'], each))) for each in _y_pred]
        except:
            print("sth is wrong")
            traceback.print_exc()
            while(1):
              pass
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
with open(os.path.join(config.CUR_PATH, "data/zy_json.json"), "r") as f:
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
    with open(os.path.join(config.CUR_PATHi, "pred_momo.txt"), 'a+')as nick:
        with open(os.path.join(config.CUR_PATHi, "ner_sample.txt"), 'a+')as f:
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

def pred_with_model_bilstm(_sentence):
    sentence = replaceNC(_sentence)
    pred = simple_cut(sentence, config.tags, y_pred_meta)
    words, clss = classifier_words(clr(_sentence), pred)
    with open("pred_with_model.txt", "a+") as f:
      f.write("*"*30)
      f.write(_sentence)
      f.write("\n1 ")
      f.write(sentence)
      f.write("\n2 ")
      f.write(",".join(words))
      f.write("\n3 ")
      f.write(",".join(pred))
      f.write("\n4 ")
      f.write(str(clss))
      f.write("\n")
      f.write("\n")
    return words , clss


