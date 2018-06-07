#coding=utf-8  
#多台机器，每台机器有一个显卡、或者多个显卡，这种训练叫做分布式训练  
import  tensorflow as tf  
#现在假设我们有A、B、C、D四台机器，首先需要在各台机器上写一份代码，并跑起来，各机器上的代码内容大部分相同  
# ，除了开始定义的时候，需要各自指定该台机器的task之外。以机器A为例子，A机器上的代码如下：  
# init the paral line struct
global NAME
global server
NAME = "server_15001"
#NAME = "server_15002"
#NAME = "server_15003"
cluster=tf.train.ClusterSpec({  
    "worker": [  
        "103.204.229.74:15001",#格式 IP地址：端口号，第1台机器的IP地址 ,在代码中需要用这台机器计算的时候，就要定义：/job:worker/task:0  
        "103.204.229.74:15002",#格式 IP地址：端口号，第2台机器的IP地址 ,在代码中需要用这台机器计算的时候，就要定义：/job:worker/task:1 
    ],  
    "ps": [  
        "103.204.229.74:15003",#格式 IP地址：端口号，第3台机器的IP地址 ,在代码中需要用这台机器计算的时候，就要定义：/job:ps/task:0  
    ]})  
if NAME == "15001":
    global server
    server = tf.train.Server(cluster,job_name='worker',task_index=0)#找到‘worker’名字下的，task0，也就是机器A
elif NAME == "15002":
    global server
    server = tf.train.Server(cluster,job_name='worker',task_index=1)#找到‘worker’名字下的，task0，也就是机器A
elif NAME == "15003":
    global server
    server = tf.train.Server(cluster,job_name='ps',task_index=0)#找到‘worker’名字下的，task0，也就是机器A
    server.join()

saver = tf.train.Saver()  
summary_op = tf.merge_all_summaries()  
init_op = tf.initialize_all_variables()  
sv = tf.train.Supervisor(init_op=init_op, summary_op=summary_op, saver=saver)  
with sv.managed_session(server.target) as sess:  
    while 1:  
        print sess.run([addwb,mutwb,divwb])  

with tf.device(tf.train.replica_device_setter(worker_device='/job:worker/task:0',cluster=cluster)):
