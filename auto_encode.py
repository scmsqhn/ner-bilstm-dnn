#coding="utf8" 
import os,sys,re,traceback,json
import pandas as pd
import numpy as np
import tensorflow as tf
#import sklearn.preprocessing as prep
#from tensorflow.examples.tutorials.mnist import input_data
import sys

def _path(filepath):
    return os.path.join(CURPATH, filepath)


def xavier_init(fan_in, fan_out, constant=1):
  low = -constant * np.sqrt(6.0 /(fan_in +fan_out))    
  high = constant * np.sqrt(6.0 /(fan_in +fan_out))    
  return tf.random_uniform((fan_in, fan_out), minval = low, maxval = high, dtype = tf.float32)

class AdditiveGaussianNoiseAutoEncoder(object):
    def __init__(self, n_input, n_hidden, transfer_function=tf.nn.softplus, optimizer=tf.train.AdamOptimizer(), scale=0.1, sess = ""):
        print('\n> cls AdditiveGaussianNoiseAutoEncoder instance')
        """
        !attention this para is used on train
        """
        self.n_samples = 200000
        self.training_epochs = 100
        self.batch_size = 32
        self.n_input = 100
        self.display_step = 1

        """
        !attention this para is used on model
        """

        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function
        self.scale = tf.placeholder(tf.float32)
        self.training_scale = scale
        network_weights = self._initialize_weights()
        self.weights = network_weights

        self.x  = tf.placeholder(tf.float32, [None, self.n_input]) 
        self.hidden = self.transfer(
            tf.add(
                tf.matmul(
                    self.x + scale * tf.random_normal((self.n_input,)),
                    self.weights['w1']), 
                self.weights['b1']
            )
        )
        self.reconstruction = tf.add(
          tf.matmul(self.hidden, self.weights['w2']), 
          self.weights['b2']
        )
        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0)) # 3 steps sub pow sum 
        self.optimizer = optimizer.minimize(self.cost)

        init = tf.global_variables_initializer()
        self.sess = sess
        #self.sess = tf.Session()
        #self.sess.run(init)
        #self.save_graph_meta()

    def save_graph_meta(self):
        tf.add_to_collection('x', self.x)
        tf.add_to_collection('scale', self.scale)
        tf.add_to_collection('cost', self.cost)
        tf.add_to_collection('reconstruction', self.reconstruction)

    def _initialize_weights(self):
       all_weights = dict()
       all_weights['w1'] = tf.Variable(xavier_init(self.n_input, self.n_hidden))
       all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype = tf.float32))
       all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden, self.n_input], dtype=tf.float32))
       all_weights['b2'] = tf.Variable(tf.zeros([self.n_input], dtype=tf.float32))
       return all_weights

    def partial_fit(self,X):
      cost,opt = self.sess.run((self.cost , self.optimizer), feed_dict={self.x:X, self.scale: self.training_scale})
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

def generat_batch(data_generate, n):
  _lst = []
  for i in range(n):
    _arr = data_generate.__next__()
    _lst.append(_arr) 
  _out = np.array(_lst).reshape(batch_size*n, n_input)
  #print(_out.shape)
  assert _out.shape == (batch_size*n,n_input)
  return _out

def gen_w2v():
   w2v = wd2vec()
   return w2v

def data_generate(w2v, df):
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
         
"""

def main():
    #--- get data
    w2v = gen_w2v()
    dg = data_generate(w2v)
    print("\n> print the df for test: ")
    print(dg.__next__())
    print(dg.__next__().shape)
    print("\n> now we going to train the model")
    
    X_train = dg
    X_test = data_generate(w2v)
    
    autoencoder = AdditiveGaussianNoiseAutoEncoder(n_input = 8,
            n_hidden = 32,
            transfer_function = tf.nn.softplus,
            optimizer = tf.train.AdamOptimizer(learning_rate = 0.001),
            scale = 0.01,
            )
    
    saver = tf.train.Saver(max_to_keep=1)
    
    model_save_path = os.path.join(CUR_PATH,'data/auto_encode.ckpt')  # 模型保存位置
    
    for epoch in range(training_epochs):
        print("\n> this is epoch", epoch)
        X_train = dg
        X_test = data_generate(w2v)
        avg_cost = 0
        total_batch = int(n_samples // batch_size)
        for i in range(total_batch):
            if i % 1000 == 1:
              print("\n> Epoch:",  '%04d' % (epoch +1), "batch ", '%04d' % (i+1))
            if i % 5000 == 1:
              _y = autoencoder.partial_predict(batch_xs)
              print("\n> Epoch:",  '%04d' % (epoch +1), "batch ", '%04d' % (i+1), "_y predict:", _y)
            batch_xs = generat_batch(X_train, 10)
            cost = autoencoder.partial_fit(batch_xs)
            avg_cost += cost / n_samples * batch_size
        if epoch % display_step == 0:
            save_path = saver.save(autoencoder.sess, model_save_path, global_step=(epoch+1))
            print("Epoch:",  '%04d' % (epoch +1), "cost=", "{:.9f}".format(avg_cost))
    
    print("Total cost: " + str(autoencoder.calc_total_cost(generat_batch(X_test, 60))))
"""    
if __name__ == "__main__":
    pass
    print("\n> autoencoder class")

