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
import neuralNetworkGeneral as nn
from Tools.inspect_checkpoint import print_tensors_in_checkpoint_file
from timeit import default_timer as timer

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
            
class Regression:
    
    def __init__(self, function, trainSize, batchSize, testSize, inputs=1, outputs=1):
        
        self.trainSize = trainSize
        self.batchSize = batchSize
        self.testSize  = testSize
        self.function  = function
        self.inputs    = inputs
        self.outputs   = outputs
        
        # make placeholders
        self.x = tf.placeholder('float', [None, inputs], name="x")
        self.y = tf.placeholder('float', [None, outputs], name="y")
        
    def generateData(self, a, b):
        
        self.xTrain, self.yTrain, self.xTest, self.yTest = \
            functionData(self.function, self.trainSize, self.testSize, a, b)
        
        
    def constructNetwork(self, nLayers, nNodes, activation='ReluSigmoid', \
                         wInitMethod='normal', bInitMethod='normal'):
        
        if activation == 'Sigmoid':
            self.neuralNetwork = lambda data : nn.modelSigmoid(self.x, nNodes=nNodes, hiddenLayers=nLayers, \
                                                               wInitMethod=wInitMethod, bInitMethod=bInitMethod)
        elif activation == 'Relu':
            self.neuralNetwork = lambda data : nn.modelRelu(self.x, nNodes=nNodes, hiddenLayers=nLayers, \
                                                            wInitMethod=wInitMethod, bInitMethod=wInitMethod)
        else:
            self.neuralNetwork = lambda data : nn.modelReluSigmoid(self.x, nNodes=nNodes, hiddenLayers=nLayers, \
                                                                   wInitMethod=wInitMethod, \
                                                                   bInitMethod=wInitMethod)
        
    def train(self, numberOfEpochs, plot=False):    
        
        trainSize = self.trainSize
        batchSize = self.batchSize
        testSize  = self.testSize 
        xTrain    = self.xTrain
        yTrain    = self.yTrain
        xTest     = self.xTest
        yTest     = self.yTest
        x         = self.x
        y         = self.y            
        
        # begin session
        with tf.Session() as sess: 
            
            # pass data to network and receive output
            self.prediction, weights, biases, neurons = self.neuralNetwork(x)
        
            cost = tf.nn.l2_loss( tf.sub(self.prediction, y) )
        
            optimizer = tf.train.AdamOptimizer().minimize(cost)
            
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
                
                """
                if epochLoss/float(trainSize) < 1e-2:
                    print 'Loss: %10g, epoch: %4d' % (epochLoss/float(trainSize), epoch)
                    break 
                """
                      
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
                        
            if plot:
                yy = sess.run(self.prediction, feed_dict={self.x: self.xTest})
                plt.plot(self.xTest[:,0], yy[:,0], 'b.')
                plt.hold('on')
                xx = np.linspace(a, b, self.testSize)
                plt.plot(xx, function(xx), 'g-')
                plt.xlabel('r')
                plt.ylabel('U(r)')
                plt.legend(['Approximation', 'L-J'])
                #plt.show()
                    
        

##### main #####

# function to approximate
function = lambda s : 1.0/s**12 - 1.0/s**6

# approximate on [a,b]
a = 0.9
b = 1.6

regress = Regression(function, int(1e6), int(1e4), int(1e3))
regress.generateData(a, b)
regress.constructNetwork(3, 10)
regress.train(20, plot=True)

                                               







