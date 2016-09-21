# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A simple MNIST classifier which displays summaries in TensorBoard.
 This is an unimpressive MNIST model, but it is a good example of using
tf.name_scope to make a graph legible in the TensorBoard graph explorer, and of
naming summary tags so that they are grouped meaningfully in TensorBoard.
It demonstrates the functionality of every TensorBoard dashboard.
"""


import tensorflow as tf
import numpy as np
import sys
import datetime as time
import os
import shutil
import matplotlib.pyplot as plt
from DataGeneration.generateData import functionData
import neuralNetworkGeneral as nn
from Tools.inspect_checkpoint import print_tensors_in_checkpoint_file
from timeit import default_timer as timer

now 			= time.datetime.now().strftime("%d.%m-%H.%M.%S")
summariesDir = 'Summaries/' + now

trainSize = int(1e6)
batchSize = int(1e4)
testSize  = batchSize

inputs  = 1
outputs = 1

a = 0.9
b = 1.6

numberOfEpochs = 10000

nNodes = 3

function = lambda s : 1.0/s**12 - 1.0/s**6

def train():
    # Import data
    xTrain, yTrain, xTest, yTest = \
        functionData(function, trainSize, testSize, a, b)
    print xTest.dtype

    sess = tf.InteractiveSession()

    # Create a multilayer model

    # Input placeholders
    with tf.name_scope('input'):
        x = tf.placeholder('float', [None, inputs],  name='x-input')
        y = tf.placeholder('float', [None, outputs], name='y-input')

    # We can't initialize these variables to 0 - the network will get stuck.
    def weight_variable(shape):
        """Create a weight variable with appropriate initialization."""
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        """Create a bias variable with appropriate initialization."""
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def variable_summaries(var, name):
        """Attach a lot of summaries to a Tensor."""
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.scalar_summary('mean/' + name, mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.scalar_summary('sttdev/' + name, stddev)
            tf.scalar_summary('max/' + name, tf.reduce_max(var))
            tf.scalar_summary('min/' + name, tf.reduce_min(var))
            """
            sess.run(tf.initialize_variables([var]))
            if name == '*/biases':
                tf.scalar_summary('value/' + name, sess.run(var[0]))
            if name == '*/weights':
                tf.scalar_summary('value/' + name, sess.run(var[0,0]))
            """
            tf.histogram_summary(name, var)           

    def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
        """Reusable code for making a simple neural net layer.
        It does a matrix multiply, bias add, and then uses relu to nonlinearize.
        It also sets up name scoping so that the resultant graph is easy to read,
        and adds a number of summary ops.
        """
        # Adding a name scope ensures logical grouping of the layers in the graph.
        with tf.name_scope(layer_name):
            # This Variable will hold the state of the weights for the layer
            with tf.name_scope('weights'):
                weights = weight_variable([input_dim, output_dim])
                variable_summaries(weights, layer_name + '/weights')
            with tf.name_scope('biases'):
                biases = bias_variable([output_dim])
                variable_summaries(biases, layer_name + '/biases')
            with tf.name_scope('Wx_plus_b'):
                preactivate = tf.matmul(input_tensor, weights) + biases
                tf.histogram_summary(layer_name + '/pre_activations', preactivate)
            if not act == None:
                activations = act(preactivate, 'activation')
                tf.histogram_summary(layer_name + '/activations', activations)
                return activations
            else:
                return preactivate

    hidden1 = nn_layer(x, 1, nNodes, 'layer1')
    hidden2 = nn_layer(hidden1, nNodes, nNodes, 'layer2', act=tf.nn.sigmoid)

    """
    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        tf.scalar_summary('dropout_keep_probability', keep_prob)
        dropped = tf.nn.dropout(hidden1, keep_prob)
    """

    prediction = nn_layer(hidden2, nNodes, 1, 'outputLayer', act=None)

    with tf.name_scope('L2Norm'):
        cost = tf.nn.l2_loss( tf.sub(prediction, y) )
        tf.scalar_summary('L2Norm', cost/batchSize)

    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer().minimize(cost)

    # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
    merged = tf.merge_all_summaries()
    train_writer = tf.train.SummaryWriter(summariesDir + '/train', sess.graph)
    test_writer = tf.train.SummaryWriter(summariesDir + '/test')
    tf.initialize_all_variables().run()


    # Train the model, and also write summaries
    # Every 10th step, measure test-set accuracy, and write test summaries
    # All other steps, run train_step on training data, & add training summaries

    # loop through epochs
    for epoch in range(numberOfEpochs):
        
        if epoch % 10 == 0:
            
            summary, c = sess.run([merged, cost], feed_dict={x: xTest, y: yTest})
            test_writer.add_summary(summary, epoch)
            print 'Cost/N at step %4d: %g' % (epoch, c/testSize)
            
            
        else:          
            i = np.random.randint(trainSize-batchSize)
            xBatch = xTrain[i:i+batchSize]
            yBatch = yTrain[i:i+batchSize]
            if epoch % 100 == 99:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, _ = sess.run([merged, train_step],
                                      feed_dict={x: xBatch, y: yBatch},
                                      options=run_options,
                                      run_metadata=run_metadata)
                train_writer.add_run_metadata(run_metadata, 'step%4d' % epoch)
                train_writer.add_summary(summary, epoch)
                print('Adding run metadata for', epoch)
                
            else:            
                summary, _ = sess.run([merged, train_step], feed_dict={x: xBatch, y: yBatch})
                train_writer.add_summary(summary, epoch)
        
    """
    def feed_dict(train):
        "Make a TensorFlow feed_dict: maps data onto Tensor placeholders."
        if train or FLAGS.fake_data:
            xs, ys = mnist.train.next_batch(100, fake_data=FLAGS.fake_data)
            k = FLAGS.dropout
        else:
            xs, ys = mnist.test.images, mnist.test.labels
            k = 1.0
        return {x: xs, y_: ys, keep_prob: k}
    
        
    for i in range(FLAGS.max_steps):
        if i % 10 == 0:  # Record summaries and test-set accuracy
            summary, acc = sess.run([merged, cost], feed_dict=feed_dict(False))
            test_writer.add_summary(summary, i)
            print('Accuracy at step %s: %s' % (i, acc))
        else:  # Record train set summaries, and train
            if i % 100 == 99:  # Record execution stats
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, _ = sess.run([merged, train_step],
                                      feed_dict=feed_dict(True),
                                      options=run_options,
                                      run_metadata=run_metadata)
                train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
                train_writer.add_summary(summary, i)
                print('Adding run metadata for', i)
            else:  # Record a summary
                summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
                train_writer.add_summary(summary, i)
    
    """
    
    train_writer.close()
    test_writer.close()


def main(_):
    train()


if __name__ == '__main__':
    tf.app.run()