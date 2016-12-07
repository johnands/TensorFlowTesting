import tensorflow as tf
import numpy as np

with tf.Session() as sess:
    #saver = tf.train.import_meta_graph('../../../TrainingData/02.12-19.36.50/Checkpoints/ckpt-16282.meta');
    #saver.restore(sess, '../../../TrainingData/02.12-19.36.50/Checkpoints/ckpt-16282')
    tf.import_graph_def('../../../TrainingData/06.12-14.19.29/graph.pb')
