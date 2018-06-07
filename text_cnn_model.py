#!coding=utf-8
import tensorflow as tf
#import numpy as np
#import logging
import pdb
#import sys
#import traceback

class TextCNN(object):
    """
    A CNN for text classification.
    embedding layer ==> convolutional ==> max-pooling == >softmax layer.
    """
    def __init__(self,
                 sequence_length,#句子长度
                 num_classes,#分类
                 vocab_size,#词向量深度
                 embedding_size,#嵌入层深度
                 filter_sizes,#卷积核大小
                 num_filters,#卷积核数
                 input_x,#输入x
                 input_y,#输入y
                 prob,#输入y
                 l2_reg_lambda=0.0):#正则惩罚项
        # Placeholders for input, output and dropout
        self.input_x = input_x#tf.placeholder(tf.int32, [None, sequence_length], name="input_x")#x输入的是n个句子
        self.input_y = input_y#tf.placeholder(tf.int32, [None, num_classes], name="input_y")#y输入的是每个句子的分类
        self.dropout_keep_prob = prob#btf.placeholder(tf.float32, name="dropout_keep_prob")
        print("input_x",input_x.shape)
        print("input_y",input_y.shape)

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)
        self.model_save_path = "./model/textcnn.ckpt"

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("att-embedding"):
            self.W_emb = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W_emb")
            self.embedded_chars = tf.nn.embedding_lookup(self.W_emb, tf.cast(self.input_x,tf.int32))
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)#增加一个维度
            print("embedded_chars",self.embedded_chars)
            print("embedded_chars_expand",self.embedded_chars_expanded)
            pdb.set_trace()

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):#有多少个卷积核尺寸就定义多少个name_scope
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("att-dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)
           

        # Final (unnormalized) scores and predictions
        with tf.name_scope("att-output" ):
            W = tf.get_variable( "Wout", shape=[num_filters_total, num_classes], initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            from_ = self.h_drop.shape
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            to_ = (self.scores)
            print(from_, "to", to_)
            print(self.scores.shape)
            #self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            #print(self.scores)
            """
            before :self.predictions is softmax
            after predictions output scores direct with no classify
            """
            self.predictions = tf.argmax(self.scores, 1, name="predictions")
            print(self.predictions.shape)
            pdb.set_trace()
            #self.predictions = self.scores
            print("\n> self.scores.shape:", self.scores.shape)
            print("\n> self.predictions.shape:", self.predictions.shape)
            pdb.set_trace()

        # Calculate mean cross-entropy loss
        with tf.name_scope("att-loss"):
            #losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=tf.cast(self.scores,tf.float64), labels=tf.cast(tf.reshape(self.input_y,[-1]),tf.float64))
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("att-accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(self.loss)
        self.train_op = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)

    def save_graph_meta_cnn(self):
        #pdb.set_trace()
        tf.add_to_collection('ix', self.input_x)
        tf.add_to_collection('iy', self.input_y)
        tf.add_to_collection('dkp', self.dropout_keep_prob)

    """
    def train(self, session_conf):
        #saver = tf.train.Saver(max_to_keep=3)
        with tf.Session() as sess:
           sess = tf.Session(config=session_conf)
           sess.run(tf.global_variables_initializer())
           #self.save_graph_meta_cnn()
           fetch = [self.train_op, self.predictions, self.accuracy]
           #feed_dict = ()
           eval_ins = Eval()
           cnt=0
           while(1):
               pdb.set_trace()
               cnt+=1
               pred_, y_batch, _y_pred_meta = eval_ins.predict()
               x_input_cnn = eval_ins.data_helper.textcnn_data_transform(_y_pred_meta, 5)
               _op, _pred, _acc = "","",""
               #pdb.set_trace()
               #_print(x_input_cnn)
               #_print(pred_)
               try:
                   _op, _pred, _acc = sess.run(fetch, feed_dict={self.input_x:x_input_cnn, self.input_y:_y_pred_meta, self.dropout_keep_prob:0.5})
               except:
                   traceback.print_exc()
               if cnt%20==1:
                   k = ["_op," "_pred", "_acc"]
                   v = [_op, _pred, _acc]
                   _print(dict(zip(k,v)))
               if cnt%10==1:
                   save_path = saver.save(sess, self.model_save_path, global_step=(cnt//100))
    """
if __name__ == "__main__":
    session_conf = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False)
    tc = TextCNN(
            sequence_length=32,\
            num_classes=18, \
            vocab_size=120000, \
            embedding_size=128,\
            num_filters=128,\
            filter_sizes=[2,3,4], \
            l2_reg_lambda=0.0)
    #tc.train(session_conf)

# END
