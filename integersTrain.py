"""
Train a neural network to compare two digits 0-9
Three classes: x1 > x2, x1 = x2 and x1 < x2: [1,0,0], [0,1,0] and [0,0,1] respectively
x: [[5,8], [9,3], [4,4], ...]
y: [[0,0,1], [1,0,0], [0,1,0], ...]
"""

import tensorflow as tf
import numpy as np
from DataGeneration.generateData import compareIntegers
import neuralNetworkModel as nn

# number of pairs to compare
N = 100000

batch_size = 100

# get data
x_train, y_train, x_test, y_test = compareIntegers(N)

# number of inputs and outputs
inputs  = 2
outputs = 3

# number of neurons in each hidden layer
noNodes = 10
nodesPerLayer = [noNodes, noNodes, noNodes]

neuralNetwork = lambda data : nn.model_1HiddenLayerRelu(data, nodesPerLayer, inputs, outputs)

x = tf.placeholder('float', [None, inputs])
y = tf.placeholder('float')

  
def train_neural_network(x):
    
    # pass data to network and receive output
    prediction = neuralNetwork(x)
    
    # define the cost/loss/error, which is the cross entropy
    # valid for mutually exclusive classes
    # prediction must be unscaledlog of probabilites
    # we do the softmax regression simultaneously for efficiency
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction, y) )
    #cost = tf.nn.l2_loss(tf.sub(prediction,y))
    
    # choose optimizer to minimize cost
    # default learning rate: 0.001
    # tf.train.AdamOptimizer is a class, optimizer is a class object
    # i.e. it is an operation that updates variables after minimizing
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    # number of cycles of feed-forward and backpropagation
    hm_epochs = 5
    
    # begin session
    with tf.Session() as sess:
        
        # run all the variable ops
        sess.run(tf.initialize_all_variables())
        
        # loop through epocs
        for epoch in range(hm_epochs):
            # track loss for each epoch
            epoch_loss = 0
            i = 0
            # loop through batches and cover whole data set for each epoch
            while i < x_train.shape[0]:
                start = i
                end   = i + batch_size
                batch_x = x_train[start:end]
                batch_y = y_train[start:end]

                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
                epoch_loss += c
                i += batch_size

            print('Epoch', epoch+1, 'completed out of', hm_epochs,'loss:', epoch_loss)
    
        # compare which class we predicted and which is correct
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        # cast converts to floats, and then we find mean, i.e.
        # fraction of correct classes
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        
        # compare to test set
        print('Accuracy:', accuracy.eval({x: x_test, y: y_test}))

##### main #####
train_neural_network(x)