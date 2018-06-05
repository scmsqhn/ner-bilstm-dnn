#coding=utf8 

import os
import sys
import re
import traceback
import json
import pandas as pd
import numpy as np
import tensorflow as tf
import sklearn.preprocessing as prep
from tensorflow.examples.tutorials.mnist import input_data
from test_request import *

library = init_arctic()
item = library.read('train_data_for_addr_classify')
item.metadata
print("*"*60)
print("!!!\tAttention this data is generate by test_request.py file and \n \
we use this to train my model, ")
print("*"*60)
print("\n>> print this to prove daat read suncc")
print(item.metadata)
print(len(item.data), "sentences num")
print(len(item.data[3]), "words num in sentence")
print(item.data[9][0].shape, "vec in word")



#DEBUG = False
DEBUG = True
CUR_PATH = os.path.dirname(__file__)
n_samples = 200000#
training_epochs = 10#
batch_size = 10#
n_input = 5000#
display_step = 1#

def xavier_init(fan_in, fan_out, constant=1):
  low = -constant * np.sqrt(6.0 /(fan_in +fan_out))    
  high = constant * np.sqrt(6.0 /(fan_in +fan_out))    
  return tf.random_uniform((fan_in, fan_out), minval = low, maxval = high, dtype = tf.float32)

class AdditiveGaussianNoiseAutoEncoder(object):
    # this is a hidden layer
    def __init__(self, n_input,n_output ,n_hidden, transfer_function=tf.nn.softplus, optimizer=tf.train.AdamOptimizer(), scale=0.1):
        self.n_input = n_input
        self.n_output = 100#n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function
        self.scale = tf.placeholder(tf.float32)
        self.training_scale = scale
        network_weights = self._initialize_weights()
        self.weights = network_weights

        self.x  = tf.placeholder(tf.float32, [None, self.n_input]) 
        self.y  = tf.placeholder(tf.float32, [None, self.n_output]) 
        
        input_2_steps = scale * tf.random_normal((self.n_input,))
        input_3_steps = self.x+input_2_steps
        input_4_steps = tf.matmul(input_3_steps, self.weights['w1'])
        self.hidden = tf.add(input_4_steps, self.weights['b1'])

        # IMPORTANT NO DEL!!!
        #self.hidden = self.transfer(tf.add(tf.matmul(
        #            self.x + scale * tf.random_normal((self.n_input,)),
        #            self.weights['w1']), 
        #    )
        #)

        # encoder-decoder reconstruction according to the result of decording
        self.reconstruction = tf.add(
          tf.matmul(self.hidden, self.weights['w2']), 
          self.weights['b2']
        )
        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.y), 2.0)) # 3 steps sub pow sum 
        # attention here self.x turn to self.y
        self.optimizer = optimizer.minimize(self.cost)

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
        tf.add_to_collection('x', self.x)
        tf.add_to_collection('scale', self.scale)
        tf.add_to_collection('cost', self.cost)
        tf.add_to_collection('reconstruction', self.reconstruction)

    def _initialize_weights(self):
       all_weights = dict()
       all_weights['w1'] = tf.Variable(xavier_init(self.n_input, self.n_hidden))
       all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype = tf.float32))
       all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden, self.n_output], dtype=tf.float32))
       all_weights['b2'] = tf.Variable(tf.zeros([self.n_output], dtype=tf.float32))
       return all_weights

    def partial_fit(self,X,Y):
      cost,opt = self.sess.run((self.cost , self.optimizer), feed_dict={self.x:X, self.y:Y, self.scale: self.training_scale})
      return cost

    def partial_predict(self,X):
      _y = self.sess.run((self.reconstruction), feed_dict={self.x:X, self.scale: self.training_scale})
      return _y

    def calc_total_cost(self,X):
      return self.sess.run(self.cost,feed_dict = {self.x:X, self.scale: self.training_scale})

    def transform(self, X):
      return self.sess.run(self.hidden, feed_dict={self.x:X, self.scale: self.training_scale})

    def generate(self, hidden=None):
      if hidden is NOne:
        hidden = np.random.normal(size=self.weights["b1"])
      return self.sess.run(self.reconstruction, feed_dict={self.hidden:hidden})

    def reconstruct(self, X):
      return self.sess.run(self.reconstrucion, feed_dict = {sekf.x:X, self.scale:self.training_scale})

    def getWeights(self):
      return self.sess.run(self.weights['w1'])

    def getBiases(self):
      return self.sess.run(self.weights['b1'])

#mnist = input_data.read_data_sets('MNIST_dat', one_hot = True)

def standard_scale(X_train, X_test):
  preprocessor = prep.StandardScaler().fit(X_train)
  X_train = preprocesor.transform(X_train)
  X_test= preprocesor.transform(X_test)
  return X_train, X_test

def get_random_block_from_data(data, batch_size):
  start_indeix = np.random.randint(0, len(data) - batch_size)
  return data[start_index:(start_index + batch_size)]

global tdg
#tdg = train_data_generate()

def generat_batch(data_generate, n):
  _lst = []
  _blst = []
  for i in range(n):
    _arr = ""
    _lb = ""
    try:
        _arr,_lb = data_generate.__next__()
    except StopIteration:
        global tdg
        tdg = train_data_generate()
        return generat_batch(tdg, n)
    _lst.append(_arr) 
    _blst.append(_lb) 
  _out = np.array(_lst).reshape(batch_size, n_input)
  _out_lb = np.array(_blst).reshape(batch_size, 100)
  #print(_out.shape)
  assert _out.shape == (batch_size,n_input)
  assert _out_lb.shape== (batch_size,100)
  return _out, _out_lb

def gen_w2v():
   import word2vec
   w2v = word2vec._gensim_word2vec.wd2vec()
   #w2v = wd2vec()
   return w2v

def train_data_generate():
    sent_len = len(item.data)
    for i in range(0,sent_len):
        sent = item.data[i]
        #print(len(sent))
        if len(sent)<3:
            continue
        label = sent[-1]
        words = sent[:-1]
        in_lst = []
        for word in words:
            in_lst.extend(list(word))
        #print(len(in_lst))
        if len(words)<50:
            ld = 50-len(words)
            in_lst.extend([0.0]*ld*100)
        #print(len(in_lst))
        assert len(in_lst)==5000
        yield np.array(in_lst), np.array(label)





def data_generate(w2v):
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

#--- get data
w2v = gen_w2v()
dg = data_generate(w2v)
print("\n> print the dg for test: ")
print(dg.__next__())
print(dg.__next__().shape)
print("\n> now we going to train the model")

X_train = dg
X_test = data_generate(w2v)

autoencoder = AdditiveGaussianNoiseAutoEncoder(n_input = 5000,
        n_hidden = 10000,
        n_output = 100,
        transfer_function = tf.nn.softplus,
        optimizer = tf.train.AdamOptimizer(learning_rate = 0.001),
        scale = 0.01)

saver = tf.train.Saver(max_to_keep=1)

model_save_path = os.path.join(CUR_PATH,'data/auto_encode.ckpt')  # 模型保存位置

for epoch in range(training_epochs):
    tdg = train_data_generate()
    print("\n> this is epoch", epoch)
    #X_train = dg
    #X_test = data_generate(w2v)
    avg_cost = 0
    total_batch = int(n_samples // batch_size)
    for i in range(total_batch):
        batch_xs, batch_ys = generat_batch(tdg,10) #X_train, 10)
        cost = autoencoder.partial_fit(batch_xs,batch_ys)
        avg_cost += cost / n_samples * batch_size
        #if i % 10 == 1:
        #  print("\n> Epoch:",  '%04d' % (epoch +1), "batch ", '%04d' % (i+1))
        if i % 50 == 1:
          _y = autoencoder.partial_predict(batch_xs)
          #print("\n> Epoch:",  '%04d' % (epoch +1), "batch ", '%04d' % (i+1), "_y predict:", _y)
          print("\n> Epoch:",  '%04d' % (epoch +1), "batch ", '%04d' % (i+1), "cost: ", cost)
          print("\n> Epoch:",  '%04d' % (epoch +1), "batch ", '%04d' % (i+1), "avg_cost: ", avg_cost)
    if epoch % display_step == 0:
        save_path = saver.save(autoencoder.sess, model_save_path, global_step=(epoch+1))
        print("Epoch:",  '%04d' % (epoch +1), "cost=", "{:.9f}".format(avg_cost))

print("Total cost: " + str(autoencoder.calc_total_cost(generat_batch(generat_batch_batch),1)[0]))

#print("Total cost: " + str(autoencoder.calc_total_cost(generat_batch(X_test, 60))))

def init_arctic(name='mongodb'):
    if name == 'mongodb':
        store = Arctic('mongodb://10.6.5.32')
        store.initialize_library('db_for_train')
        library = store['db_for_train']
    return library

if __name__ == "__main__":
    pass
    if DEBUG:
      print("\n>> autoencoder")
    print("\n>> test_request.py")
    #mongoclient = pymongo.MongoClient("mongodb://10.6.5.32:27017")
    #db=mongoclient['myDB']
    #train_data_coll = db['tmp_train_adr_classify']
    #item.data

