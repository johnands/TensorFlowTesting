"""
Train a neural network to approximate a continous function
"""

import tensorflow as tf
import numpy as np
import sys
import datetime as time
import os
import shutil
import matplotlib.pyplot as plt
from DataGeneration.generateData import functionData
import neuralNetworkModel as nn
import neuralNetworkXavier as nnx
from Tools.inspect_checkpoint import print_tensors_in_checkpoint_file

loadFlag        = False
loadFileName    = ''
saveFlag        = False
saveDirName     = ''
saveMetaName    = ''
trainingDir     = 'TrainingData'

# process command-line input
if len(sys.argv) > 1:
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == '--load':
            i = i + 1
            loadFlag     = True
            loadFileName = sys.argv[i]
        elif sys.argv[i] == '--save':
            i = i + 1
            saveFlag = True
            now 			= time.datetime.now().strftime("%d.%m-%H.%M.%S")
            saveDirName 	= trainingDir + '/' + now
            saveMetaName	= saveDirName + '/' + 'meta.dat'

            # If this directory exists
            if os.path.exists(saveDirName) :
                print "Attempted to place data in existing directory, %s. Exiting." % \
                (saveDirName)
                exit(1)
            else:
                os.mkdir(saveDirName)

            # Copy the python source code used to run the training, to preserve
            # the tf graph (which is not saved by tf.nn.Saver.save()).
            shutil.copy2(sys.argv[0], saveDirName + '/')

        else:
            i = i + 1

# reset so that variables are not given new names
tf.reset_default_graph()

# number of samples
trainSize = int(1e6)

# function to approximate
function = lambda s : 1.0/s**12 - 1.0/s**6

batchSize = int(1e4)
testSize = int(1e3)

# get random input
a = 0.9
b = 1.6
xTrain, yTrain, xTest, yTest = functionData(function, trainSize, testSize, a, b)

# number of inputs and outputs
inputs  = 1
outputs = 1

# number of neurons in each hidden layer
nNodes = 10
#nodesPerLayer = [noNodes, noNodes, noNodes]
hiddenLayers = 3

#neuralNetwork = lambda data : nn.model_1HiddenLayerSigmoid(data, nodesPerLayer, inputs, outputs)
neuralNetwork = lambda data : nnx.modelSigmoid(data, nNodes=nNodes, hiddenLayers=hiddenLayers,
                                               wInitMethod='normal', bInitMethod='normal')

x = tf.placeholder('float', [None, inputs], name="x")
y = tf.placeholder('float', [None, outputs], name="y")

#print_tensors_in_checkpoint_file("SavedNetworks/func.ckpt", None)

  
def train_neural_network(x, plot=False):
    
    # begin session
    with tf.Session() as sess: 
        
        # pass data to network and receive output
        #prediction = neuralNetwork(x)
        prediction, weights, biases, neurons = neuralNetwork(x)
    
        cost = tf.nn.l2_loss( tf.sub(prediction, y) )
    
        optimizer = tf.train.AdamOptimizer().minimize(cost)
    
        # number of cycles of feed-forward and backpropagation
        numberOfEpochs = 20
        
        # initialize variables or restore from file
        saver = tf.train.Saver(weights + biases, max_to_keep=None)
        sess.run(tf.initialize_all_variables())
        if loadFlag:
            saver.restore(sess, loadFileName)
            print 'Model restored'

        
        # loop through epocs
        for epoch in range(numberOfEpochs):
            # track loss for each epoch
            epochLoss = 0
            i = 0
            # loop through batches and cover whole data set for each epoch
            while i < trainSize:
                start = i
                end   = i + batchSize
                batchX = xTrain[start:end]
                batchY = yTrain[start:end]

                _, c = sess.run([optimizer, cost], feed_dict={x: batchX, y: batchY})
                epochLoss += c
                i += batchSize
                
            # compute test set loss
            _, testCost = sess.run([optimizer, cost], feed_dict={x: xTest, y: yTest})
            
            print 'Epoch %5d completed out of %5d loss/N: %15g' % \
                  (epoch+1, numberOfEpochs, epochLoss/trainSize)
                  
            # If saving is enabled, save the graph variables ('w', 'b') and dump
            # some info about the training so far to SavedModels/<this run>/meta.dat.
            if saveFlag:
                if epoch == 0:
                    saveEpochNumber = 0
                    with open(saveMetaName, 'w') as outFile:
                        outStr = '# epochs: %d train: %d, test: %d, batch: %d, nodes: %d, layers: %d' % \
                                  (numberOfEpochs, trainSize, testSize, batchSize, nNodes, hiddenLayers)
                        outFile.write(outStr + '\n')
                else:
                    with open(saveMetaName, 'a') as outFile:
                        outStr = '%g %g' % (epochLoss/float(trainSize), testCost/float(testSize))
                        outFile.write(outStr + '\n')

                if epoch % 10 == 0:
                    saveFileName = saveDirName + '/' 'ckpt'
                    saver.save(sess, saveFileName, global_step=saveEpochNumber)
                    saveEpochNumber = saveEpochNumber + 1
        
        
        # plot prediction together with correct function  
        # how well does the neural network approximate the function
        # on the test data after all epochs are done?
        if plot:
            yy = sess.run(prediction, feed_dict={x: xTest})
            plt.plot(xTest[:,0], yy[:,0], 'b.')
            plt.hold('on')
            xx = np.linspace(a, b, testSize)
            plt.plot(xx, function(xx), 'g-')
            plt.xlabel('r')
            plt.ylabel('U(r)')
            plt.legend(['Approximation', 'L-J'])
            #plt.show()

    return weights, biases, neurons
    
      

##### main #####
weights, biases, neurons = train_neural_network(x, plot=True)