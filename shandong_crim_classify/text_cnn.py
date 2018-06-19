#!coding=utf-8

import tensorflow as tf
import numpy as np

DEBUG = True

class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """

    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.int32, [None], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
        import pdb
        if DEBUG:
            pdb.set_trace()

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, 1]
                #filter_shape = [filter_size, embedding_size, 1, num_filters]
                print('filter_shape', filter_shape)
                import pdb
                if DEBUG:
                    pdb.set_trace()
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                if DEBUG:
                    # Apply nonlinearity
                    pdb.set_trace()
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.avg_pool(
                    h,
                    ksize=[1, filter_size + 1, 1, 1],
                    #ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                if DEBUG:
                    pdb.set_trace()
                pooled_outputs.append(pooled)
        pooled_outputs.append(tf.reshape(self.embedded_chars_expanded, (32,200*128,1,1)))

        # Combine all the pooled features
        num_filters_total = 197+195+200*128#200+198+197#num_filters * len(filter_sizes)
        #self.h_pool_flat = pooled_outputs
        self.h_pool = tf.concat(pooled_outputs, 1)
        if DEBUG:
            pdb.set_trace()
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self._scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            if DEBUG:
                pdb.set_trace()
            self.scores = tf.cast(tf.reshape(self._scores, [32,18]), tf.float32)
            self.predictions = tf.argmax(self.scores, 1, name='predictions')
            #self.predictions = tf.cast(tf.argmax(tf.cast(self.scores, tf.float32), 1), tf.int32, name="predictions")
            #self.predictions = tf.arg_max(self.scores, 1, name="predictions")

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = tf.reshape(self.input_y, [-1]), logits = self.scores)) #self.attention))
            #self.TP, self.FP, self.TN, self.FN, self.Precision, self.Recall, self.floss = self.fscore(tf.reshape(self.input_y, [-1]), self.scores) #self.attention))
            #losses = tf.nn.softmax_cross_entropy_with_logits(logits=tf.clip_by_value(self.scores, 0, 1), labels=tf.cast(tf.one_hot(self.input_y, 18), tf.float32))
            #losses = tf.nn.softmax_cross_entropy_with_logits(logits=tf.cast(tf.one_hot(tf.argmax(self.scores, 1), 18),tf.float32), labels=tf.cast(tf.one_hot(self.input_y, 18), tf.float32))
            #self.losses = tf.reduce_mean(losses)
            #self.loss = tf.reduce_mean(losses)# + l2_reg_lambda * l2_loss
            #self.loss = losses
            #self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss
            #return TP, FP, TN, FN, Precision, Recall, floss

        # Accuracy
        with tf.name_scope("accuracy"):
            print("predictions", self.predictions.shape)
            print("input_y",self.input_y)
            correct_predictions = tf.equal(tf.cast(self.predictions, tf.int32), tf.cast(self.input_y, tf.int32))
            print(correct_predictions)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
            #self.cost= tf.cast(self.accuracy, tf.float32)
            self.correct_predictions = correct_predictions
            self.loss = self.loss  * (10 / (self.accuracy+0.01))


    def fscore(self, labels, logits): # y y_
        y = tf.cast(tf.reshape(labels, [-1]), tf.int32)

        y_ = tf.cast(tf.argmax(logits, 1), tf.int32)
        if DEBUG:
            import pdb
            pdb.set_trace()
            pass
        ylbEqPred = tf.cast(tf.equal(y,y_),tf.int32)
        ylbEqZero = tf.cast(tf.equal(y,0),tf.int32)
        ylbGthZero = tf.cast(tf.greater(y,0),tf.int32)
        yPredEqZero = tf.cast(tf.equal(y_, 0),tf.int32)
        yPredGthZero = tf.cast(tf.greater(y_, 0),tf.int32)
        TP = tf.reduce_sum(tf.cast(tf.where(tf.greater(yPredGthZero,0*yPredGthZero),ylbGthZero, 0*yPredGthZero), tf.float32))
        FP = tf.reduce_sum(tf.cast(tf.where(tf.greater(yPredGthZero,0),ylbEqZero,0*yPredGthZero), tf.float32))
        TN = tf.reduce_sum(tf.cast(tf.where(tf.greater(yPredEqZero,0),ylbEqZero, 0*yPredEqZero), tf.float32))
        FN = tf.reduce_sum(tf.cast(tf.where(tf.greater(yPredEqZero,0),ylbGthZero,0*yPredGthZero), tf.float32))
        TP = tf.add(TP,1)
        TN = tf.add(TN,1)
        FP = tf.add(FP,1)
        FN = tf.add(FN,1)
        Precision = tf.divide(TP, tf.add(TP,FP))
        Recall= tf.divide(TP, tf.add(TP,FN))
        Fscore = tf.divide(tf.multiply(Precision, Recall), tf.add(Precision, Recall))
        squareFscore = tf.multiply(Fscore, Fscore)
        threeMulFscore = tf.multiply(squareFscore, Fscore)
        forMulFscore = tf.multiply(threeMulFscore, Fscore)
        #fourMulFscore = tf.multiply(threeMulFscore, Fscore) # if u wanna fscore more effection modify here
        """ this to make the fscore more important before close enought to zero """
        floss= tf.divide(10, forMulFscore)
        return TP, FP, TN, FN, Precision, Recall, floss



