import numpy as np 
import timer
from keras.datasets import mnist 
from keras.models import Model #泛型模型 
from keras.layers import Dense, Input 
import matplotlib.pyplot as plt 

_tc = timer.TimeCnt()
_tc.cnt_time()

_dic = dict()

#def time_cnt(func):  
#    def new_func(a, b):  
#        _TimeCnt = TimeCnt()
#        #_TimeCnt.cnt_time()
#        result = func(a, b)  
#        _TimeCnt.cnt_time()
#        #print "result:", result, "used:", (end_tiem - start_time).microseconds, "μs"  
#        #return result
#    return new_func  

np.random.seed(1337) # for reproducibility 

_tc.cnt_time()
# X shape (60,000 28x28), y shape (10,000, ) 
(x_train, _), (x_test, y_test) = mnist.load_data() 

_tc.cnt_time()
# 数据预处理 
x_train = x_train.astype('float32') / 255. - 0.5 # minmax_normalized 
x_test = x_test.astype('float32') / 255. - 0.5 # minmax_normalized 
x_train = x_train.reshape((x_train.shape[0], -1)) 
x_test = x_test.reshape((x_test.shape[0], -1)) 
print(x_train.shape) 
print(x_test.shape) 

_tc.cnt_time()
# 压缩特征维度至2维 
encoding_dim = 2 

# this is our input placeholder 
input_img = Input(shape=(784,)) 

# 编码层 
encoded = Dense(128, activation='relu')(input_img) 
encoded = Dense(64, activation='relu')(encoded) 
encoded = Dense(10, activation='relu')(encoded) 
encoder_output = Dense(encoding_dim)(encoded) 

# 解码层 
decoded = Dense(10, activation='relu')(encoder_output) 
decoded = Dense(64, activation='relu')(decoded) 
decoded = Dense(128, activation='relu')(decoded) 
decoded = Dense(784, activation='tanh')(decoded) 

# 构建自编码模型 
autoencoder = Model(inputs=input_img, outputs=decoded) 

# 构建编码模型 
encoder = Model(inputs=input_img, outputs=encoder_output) 

# compile autoencoder 
autoencoder.compile(optimizer='adam', loss='mse') 

_tc.cnt_time()
# training 
autoencoder.fit(x_train, x_train, epochs=1, batch_size=256, shuffle=True) 

_tc.cnt_time()
# plotting 
encoded_imgs = encoder.predict(x_test) 
plt.scatter(encoded_imgs[:, 0], encoded_imgs[:, 1], c=y_test, s=3) 
#plt.scatter(encoded_imgs[:, 0], encoded_imgs[:, 1], c=_, s=3) 
plt.colorbar() 
_tc.cnt_time()
plt.show() 



