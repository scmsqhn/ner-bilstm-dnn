#
import tensorflow as tf
sess = tf.Session()
graph = tf.Graph()
save = tf.train.import_meta_graph("/home/siyuan/data/model/ckpt-1.meta")
print(graph.collections)
print(tf.global_variables)

