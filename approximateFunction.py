"""
Train a neural network to approximate a continous function
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from DataGeneration.generateData import functionData
import neuralNetworkModel as nn
import neuralNetworkXavier as nnx
from Tools.inspect_checkpoint import print_tensors_in_checkpoint_file

# number of samples
N = int(1e6)

# file name to store network
filename = "SavedNetworks/func2.ckpt"

# function to approximate
function = lambda s : 1.0/s**12 - 1.0/s**6

batch_size = int(1e4)
test_size = batch_size

# get data
a = 0.9
b = 1.6
x_train, y_train, x_test, y_test = functionData(function, N, a, b)

# number of inputs and outputs
inputs  = 1
outputs = 1

# number of neurons in each hidden layer
noNodes = 10
nodesPerLayer = [noNodes, noNodes, noNodes]
hiddenLayers = 3

#neuralNetwork = lambda data : nn.model_1HiddenLayerSigmoid(data, nodesPerLayer, inputs, outputs)
neuralNetwork = lambda data : nnx.model(data, noNodes, hiddenLayers)

x = tf.placeholder('float', [None, inputs], name="x")
y = tf.placeholder('float', [None, outputs], name="y")

#print_tensors_in_checkpoint_file("SavedNetworks/func.ckpt", None)

  
def train_neural_network(x, plot=False):
    
    # pass data to network and receive output
    #prediction = neuralNetwork(x)
    prediction, weights, biases, neurons = neuralNetwork(x)
    
    cost = tf.nn.l2_loss( tf.sub(prediction, y) )
    

    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    # number of cycles of feed-forward and backpropagation
    hm_epochs = 20
    
    # save state
    #saver = tf.train.Saver(weights + biases)
    
    # begin session
    with tf.Session() as sess:
        
        # run all the variable ops
        sess.run(tf.initialize_all_variables())
        #saver.restore(sess, filename)
        
        # loop through epocs
        for epoch in range(hm_epochs):
            # track loss for each epoch
            epoch_loss = 0
            i = 0
            # loop through batches and cover whole data set for each epoch
            while i < N:
                start = i
                end   = i + batch_size
                batch_x = x_train[start:end]
                batch_y = y_train[start:end]

                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
                epoch_loss += c
                i += batch_size
                
            # compute test set loss
            #_, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
            
            print 'Epoch %5d completed out of %5d loss/N: %15g' % \
                  (epoch+1, hm_epochs, epoch_loss/N)
    
        
        #saver.save(sess, filename)
        
        
        # plot prediction and correct function
        if plot:
            testBatch = np.random.choice(x_test[:,0], size=[test_size,1])
            yy = sess.run(prediction, feed_dict={x: testBatch})
            plt.plot(testBatch[:,0], yy[:,0], 'b.')
            plt.hold('on')
            xx = np.linspace(a, b, N)
            plt.plot(xx, function(xx), 'g-')
            #plt.show()
        
            batch_x = x_train[0:4]
            batch_y = y_train[0:4]
        
            print "hei3h"
            print "..."
        
            print "x=", batch_x
            print "y=", batch_y
            print "net(x)=", sess.run(prediction, feed_dict={x: batch_x, y: batch_y})
        

    return weights, biases, neurons
        
      

##### main #####
weights, biases, neurons = train_neural_network(x)