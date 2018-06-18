import pdb
import tensorflow as tf
import numpy as np

sess = tf.Session()

#filter_size = [2,2,1,2]

Base = np.array([ i for i in range(10000)]).reshape(1,100,100,1)

a = tf.concat([Base,Base],1)
b = tf.concat([Base,Base],2)
c = tf.concat([Base,Base],3)

pdb.set_trace()

W = tf.constant(2,shape=[10,10,1,2])

conv = tf.nn.conv2d(
    tf.cast(Base,tf.float32),
    tf.cast(W,tf.float32),
    strides=[1, 1, 1, 1],
    padding="VALID",
    name="conv")
sess.run(tf.global_variables_initializer())
res = sess.run(conv)
pdb.set_trace()

#ksize = tf.constant(2,shape=[99,99,1,99])
pooled = tf.nn.max_pool(
    tf.cast(Base,tf.float32),
    ksize=[1, 3, 3, 1],
    strides=[1, 1, 1, 1],
    padding='VALID',
    name="pool")

pool = sess.run(pooled)
pdb.set_trace()



