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
import neuralNetworkClass as nn
#import neuralNetworkGeneral as nn
from Tools.inspect_checkpoint import print_tensors_in_checkpoint_file
from timeit import default_timer as timer

loadFlag        = False
loadFileName    = ''
saveFlag        = False
saveDirName     = ''
summaryFlag     = False
summaryDir      = ''
saveMetaName    = ''
trainingDir     = 'TrainingData'

# process command-line input
if len(sys.argv) > 1:
    i = 1
    while i < len(sys.argv):
        
        if sys.argv[i] == '--load':
            i += 1
            loadFlag     = True
            loadFileName = sys.argv[i]
            
        elif sys.argv[i] == '--save':
            i += 1
            saveFlag = True
            
            # make new directory if not already made
            if saveDirName == '':
                now 			= time.datetime.now().strftime("%d.%m-%H.%M.%S")
                saveDirName 	= trainingDir + '/' + now
                
                # If this directory exists
                if os.path.exists(saveDirName) :
                    print "Attempted to place data in existing directory, %s. Exiting." % saveDirName
                    exit(1)
                else:
                    os.mkdir(saveDirName + '/checkpoints')
                    
            saveMetaName	= saveDirName + '/' + 'meta.dat'

            # Copy the python source code used to run the training, to preserve
            # the tf graph (which is not saved by tf.nn.Saver.save()).
            shutil.copy2(sys.argv[0], saveDirName + '/')
            
        elif sys.argv[i] == '--summary':
            i += 1
            summaryFlag  = True   
            if saveDirName == '':
                now 			= time.datetime.now().strftime("%d.%m-%H.%M.%S")
                saveDirName 	= trainingDir + '/' + now
                
                # If this directory exists
                if os.path.exists(saveDirName) :
                    print "Attempted to place data in existing directory, %s. Exiting." % \
                    (saveDirName)
                    exit(1)
                else:
                    os.mkdir(saveDirName + '/summaries')
                    
        else:
            i += 1
            
class Regression:
    
    def __init__(self, function, trainSize, batchSize, testSize, inputs=1, outputs=1):
        
        self.trainSize = trainSize
        self.batchSize = batchSize
        self.testSize  = testSize
        self.function  = function
        self.inputs    = inputs
        self.outputs   = outputs


    def generateData(self, a, b):
        
        self.xTrain, self.yTrain, self.xTest, self.yTest = \
            functionData(self.function, self.trainSize, self.testSize, a, b)
        
        
    def constructNetwork(self, nLayers, nNodes, activation='ReluSigmoid', \
                         wInitMethod='normal', bInitMethod='normal'):
                             
        self.nLayers = nLayers
        self.nNodes  = nNodes
                      
        # Input placeholders
        with tf.name_scope('input'):
            self.x = tf.placeholder('float', [None, self.inputs],  name='x-input')
            self.y = tf.placeholder('float', [None, self.outputs], name='y-input')

        
        self.neuralNetwork = nn.neuralNetwork(nNodes, nLayers, weightsInit=wInitMethod, biasesInit=bInitMethod,
                                              stdDev=1.0)
        self.makeNetwork = lambda data : self.neuralNetwork.modelSigmoid(self.x)
        
        
        """if activation == 'Sigmoid':
            self.neuralNetwork = lambda data : nn.modelSigmoid(self.x, nNodes=nNodes, hiddenLayers=nLayers, \
                                                               wInitMethod=wInitMethod, bInitMethod=bInitMethod)
        elif activation == 'Relu':
            self.neuralNetwork = lambda data : nn.modelRelu(self.x, nNodes=nNodes, hiddenLayers=nLayers, \
                                                            wInitMethod=wInitMethod, bInitMethod=wInitMethod)
        else:
            self.neuralNetwork = lambda data : nn.modelReluSigmoid(self.x, nNodes=nNodes, hiddenLayers=nLayers, \
                                                                   wInitMethod=wInitMethod, \
                                                                   bInitMethod=wInitMethod)"""
    
    
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
        nNodes    = self.nNodes
        nLayers   = self.nLayers        
        
        # begin session
        with tf.Session() as sess: 
            
            # pass data to network and receive output
            self.prediction = self.makeNetwork(x)
            #self.prediction, weights, biases, neurons = self.neuralNetwork(x)
            
            with tf.name_scope('L2Norm'):
                cost = tf.nn.l2_loss( tf.sub(self.prediction, y) )
                tf.scalar_summary('L2Norm', cost/batchSize)

            with tf.name_scope('train'):
                trainStep = tf.train.AdamOptimizer().minimize(cost)
            
            # initialize variables or restore from file
            saver = tf.train.Saver(self.neuralNetwork.allWeights + self.neuralNetwork.allBiases, 
                                   max_to_keep=None)
            sess.run(tf.initialize_all_variables())
            if loadFlag:
                saver.restore(sess, loadFileName)
                print 'Model restored'
                
            # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
            merged = tf.merge_all_summaries()
            train_writer = tf.train.SummaryWriter(summaryDir + '/train', sess.graph)
            test_writer = tf.train.SummaryWriter(summaryDir + '/test')
            tf.initialize_all_variables().run()
                    
            # train
            for epoch in xrange(numberOfEpochs):
                
                # pick random batches
                i = np.random.randint(trainSize-batchSize)
                xBatch = xTrain[i:i+batchSize]
                yBatch = yTrain[i:i+batchSize]
                
                # calculate cost on test set every 10th epoch
                if epoch % 10 == 0:               
                    summary, testCost = sess.run([merged, cost], feed_dict={x: xTest, y: yTest})
                
                # run and write meta data every 100th step
                if epoch % 100 == 99:
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    summary, _ = sess.run([merged, trainStep],
                                          feed_dict={x: xBatch, y: yBatch},
                                          options=run_options,
                                          run_metadata=run_metadata)
                    train_writer.add_run_metadata(run_metadata, 'step%4d' % epoch)
                    train_writer.add_summary(summary, epoch)
                    print('Adding run metadata for', epoch)    
                
                # do ordinary train step else
                else:
                    summary, trainCost, _ = sess.run([merged, cost, trainStep], feed_dict={x: xBatch, y: yBatch}) 
                    
                    
                    
                    test_writer.add_summary(summary, epoch)
                    print 'Cost/N at step %4d: %g' % (epoch, epochCost/testSize)                 
                    
                else:          

                    

                        
                    else:            
                        summary, _ = sess.run([merged, trainStep], feed_dict={x: xBatch, y: yBatch})
                        train_writer.add_summary(summary, epoch)
                
                      
                # If saving is enabled, save the graph variables ('w', 'b') and dump
                # some info about the training so far to SavedModels/<this run>/meta.dat.
                if saveFlag:
                    if epoch == 0:
                        saveEpochNumber = 0
                        with open(saveMetaName, 'w') as outFile:
                            outStr = '# epochs: %d train: %d, test: %d, batch: %d, nodes: %d, layers: %d' % \
                                      (numberOfEpochs, trainSize, testSize, batchSize, nNodes, nLayers)
                            outFile.write(outStr + '\n')
                    else:
                        with open(saveMetaName, 'a') as outFile:
                            outStr = '%g %g' % (epochLoss/float(trainSize), testCost/float(testSize))
                            outFile.write(outStr + '\n')
    
                    if epoch % 10 == 0:
                        saveFileName = saveDirName + '/' 'ckpt'
                        saver.save(sess, saveFileName, global_step=saveEpochNumber)
                        saveEpochNumber = saveEpochNumber + 1
            """            
            if plot:
                yy = sess.run(self.prediction, feed_dict={self.x: self.xTest})
                plt.plot(self.xTest[:,0], yy[:,0], 'b.')
                plt.hold('on')
                xx = np.linspace(a, b, self.testSize)
                plt.plot(xx, function(xx), 'g-')
                plt.xlabel('r')
                plt.ylabel('U(r)')
                plt.legend(['Approximation', 'L-J'])
                #plt.show()"""
                    
        

##### main #####

def performanceTest(maxEpochs, maxLayers, maxNodes):
    
    # function to approximate
    function = lambda s : 1.0/s**12 - 1.0/s**6
    
    # approximate on [a,b]
    a = 0.9
    b = 1.6

    regress = Regression(function, int(1e6), int(1e4), int(1e3))
    regress.generateData(a, b)
    
    # finding optimal value
    counter = 0
    for layers in range(1, maxLayers+1, 1):
        for nodes in range(layers, maxNodes+1, 1):
            start = timer()
            regress.constructNetwork(layers, nodes)
            regress.train(maxEpochs)
            end = timer()
            timeElapsed = end - start
            print "Layers: %2d, nodes: %2d, time = %10g" % (layers, nodes, timeElapsed)
            print
    
            if counter == 0:
                with open('Tests/timeElapsed', 'w') as outFile:
                    outStr = "Timing analysis"
                    outFile.write(outStr + '\n')
                    
            with open('Tests/timeElapsed', 'a') as outFile:
                outStr = "Layers: %2d, nodes: %2d, time = %10g" % (layers, nodes, timeElapsed)
                outFile.write(outStr + '\n')
            
            counter += 1
    
    
function = lambda s : 1.0/s**12 - 1.0/s**6
regress = Regression(function, int(1e6), int(1e4), int(1e3))
regress.generateData(0.9, 1.6)
regress.constructNetwork(3, 5, activation='Sigmoid', wInitMethod='normal', bInitMethod='normal')
regress.train(200)


                                               







