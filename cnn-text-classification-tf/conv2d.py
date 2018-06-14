import pdb
import tensorflow as tf
import numpy as np

Base = np.array([i*1.0 for i in range(10000)]).reshape(1,100,100,1)
W = np.array([i*1.0 for i in range(2)]).reshape(2,1,1,1)

conv = tf.nn.conv2d(
    Base,
    W,
    strides=[1, 1, 1, 1],
    padding="VALID",
    name="conv")
sess = tf.Session()
res = sess.run(conv)
print(res.shape)
pdb.set_trace()




