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
from Tools.inspect_checkpoint import print_tensors_in_checkpoint_file
from timeit import default_timer as timer

loadFlag        = False
loadFileName    = ''
saveFlag        = False
saveDirName     = ''
summaryFlag     = False
summaryDir      = ''
saveGraphFlag   = False
saveMetaName    = ''
now             = time.datetime.now().strftime("%d.%m-%H.%M.%S")
trainingDir     = 'TrainingData' + '/' + now

# make directory for training data
if len(sys.argv) > 1:
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == '--save' or sys.argv[i] == '--summary' == or sys.argv[i] == 'savegraph':
            if os.path.exists(trainingDir):
                print "Attempted to place data in existing directory, %s. Exiting." % trainingDir
                exit(1)
            else:
                os.mkdir(trainingDir)
                saveMetaName = trainingDir + '/' + 'meta.dat'
                break
        i += 1
        

# process command-line input
if len(sys.argv) > 1:
    i = 1
    while i < len(sys.argv):
        
        if sys.argv[i] == '--load':
            i += 1
            loadFlag     = True
            loadFileName = sys.argv[i]
            i += 1
            
        elif sys.argv[i] == '--save':
            i += 1
            saveFlag = True
            
            # make new directory for checkpoints
            saveDirName 	= trainingDir + '/Checkpoints'
            os.mkdir(saveDirName)

            # Copy the python source code used to run the training, to preserve
            # the tf graph (which is not saved by tf.nn.Saver.save()).
            shutil.copy2(sys.argv[0], saveDirName + '/')
            
        elif sys.argv[i] == '--summary':
            i += 1
            summaryFlag  = True
            
            # make new directory for summaries
            summaryDir = trainingDir + '/Summaries'
            os.mkdir(summaryDir)
            
        elif sys.argv[i] == '--savegraph':
            i += 1
            saveGraphFlag = True                    
                    
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
        
        self.a, self.b = a, b        
        self.xTrain, self.yTrain, self.xTest, self.yTest = \
            functionData(self.function, self.trainSize, self.testSize, a, b)
        
        
    def constructNetwork(self, nLayers, nNodes, activation=tf.nn.relu, \
                         wInit='normal', bInit='normal'):
                             
        self.nLayers = nLayers
        self.nNodes  = nNodes
        self.activation = activation
        self.wInit = wInit
        self.bInit = bInit
                      
        # input placeholders
        with tf.name_scope('input'):
            self.x = tf.placeholder('float', [None, self.inputs],  name='x-input')
            self.y = tf.placeholder('float', [None, self.outputs], name='y-input')

        
        self.neuralNetwork = nn.neuralNetwork(nNodes, nLayers, activation,
                                              weightsInit=wInit, biasesInit=bInit,
                                              stdDev=1.0)
        self.makeNetwork = lambda data : self.neuralNetwork.model(self.x)
        
        
        """if activation == 'Sigmoid':
            self.makeNetwork = lambda data : self.neuralNetwork.modelSigmoid(self.x)
        elif activation == 'Relu':
            self.makeNetwork = lambda data : self.neuralNetwork.modelSigmoid(self.x)
        else:
            self.makeNetwork = lambda data : self.neuralNetwork.modelReluSigmoid(self.x)"""
    
    
    def train(self, numberOfEpochs, plot=False, counter=0):    
        
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
            prediction = self.makeNetwork(x)
            #self.prediction, weights, biases, neurons = self.neuralNetwork(x)
            
            with tf.name_scope('L2Norm'):
                cost = tf.nn.l2_loss( tf.sub(prediction, y) )
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
            if summaryFlag:
                merged = tf.merge_all_summaries()
                train_writer = tf.train.SummaryWriter(summaryDir + '/train', sess.graph)
                test_writer = tf.train.SummaryWriter(summaryDir + '/test')
                    
            # train
            for epoch in xrange(numberOfEpochs):
                
                # pick random batches
                i = np.random.randint(trainSize-batchSize)
                xBatch = xTrain[i:i+batchSize]
                yBatch = yTrain[i:i+batchSize]
                           
                # calculate cost on test set every 10th epoch
                if epoch % 100 == 0:               
                    testCost = sess.run(cost, feed_dict={x: xTest, y: yTest})
                    print 'Cost/N at step %4d: %g' % (epoch, testCost/testSize)
                    if summaryFlag:
                        summary = sess.run(merged, feed_dict={x: xTest, y: yTest})
                        test_writer.add_summary(summary, epoch)
                
                """
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
                """
                
                # train      
                trainCost, _ = sess.run([cost, trainStep], feed_dict={x: xBatch, y: yBatch})
                if summaryFlag:
                    if epoch % 10 == 0:
                        summary = sess.run(merged, feed_dict={x: xBatch, y: yBatch})
                        train_writer.add_summary(summary, epoch)
                                                        
                # If saving is enabled, save the graph variables ('w', 'b') and dump
                # some info about the training so far to SavedModels/<this run>/meta.dat.
                                  
                
                if saveFlag or summaryFlag:
                    if epoch == 0:
                        saveEpochNumber = 0
                        with open(saveMetaName + '%1d' % counter, 'w') as outFile:
                            outStr = """# epochs: %d train: %d, test: %d, batch: %d, nodes: %d, layers: %d
                                        a: %1.1f, b: %1.1f, activation: %s, wInit: %s, bInit: %s""" % \
                                      (numberOfEpochs, trainSize, testSize, batchSize, nNodes, nLayers, \
                                       self.a, self.b, self.activation, self.wInit, self.bInit)
                            outFile.write(outStr + '\n')
                            outStr = '%g %g' % (trainCost/float(batchSize), testCost/float(testSize))
                            outFile.write(outStr + '\n')
                    else:
                        if epoch % 10 == 0:
                             with open(saveMetaName, 'a') as outFile :
                                 outStr = '%g %g' % (trainCost/float(batchSize), testCost/float(testSize))
                                 outFile.write(outStr + '\n')                   
                    
                """if saveFlag: 
                    if epoch % 100 == 0:
                        saveFileName = saveDirName + '/' 'ckpt'
                        saver.save(sess, saveFileName, global_step=saveEpochNumber)
                        saveEpochNumber += 1"""
       
            # write weights and biases to file when training is finished
            with open('saveNetwork.txt', 'w') as outFile:
                outStr = "%1d %1d %s" % (nLayers, nNodes, self.activation.__name__)
                outFile.write(outStr + '\n')
                size = len(self.neuralNetwork.allWeights)
                for i in range(size):
                    weights = sess.run(self.neuralNetwork.allWeights[i])
                    if i < size-1:
                        for j in range(len(weights)):
                            for k in range(len(weights[0])):
                                outFile.write("%g" % weights[j][k])
                                outFile.write(" ")
                            outFile.write("\n")
                    else:
                        for j in range(len(weights[0])):
                            for k in range(len(weights)):
                                outFile.write("%g" % weights[k][j])
                                outFile.write(" ")
                            outFile.write("\n")
                        
                outFile.write("\n")
                    
                for biasVariable in self.neuralNetwork.allBiases:
                    biases = sess.run(biasVariable)
                    for j in range(len(biases)):
                        outFile.write("%g" % biases[j])
                        outFile.write(" ")
                    outFile.write("\n")
                      
            if plot:
                yy = sess.run(prediction, feed_dict={self.x: self.xTest})
                plt.plot(self.xTest[:,0], yy[:,0], 'b.')
                plt.hold('on')
                xx = np.linspace(self.a, self.b, self.testSize)
                plt.plot(xx, self.function(xx), 'g-')
                plt.xlabel('r')
                plt.ylabel('U(r)')
                plt.legend(['Approximation', 'L-J'])
                plt.show()
                    
        

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
            
def testActivations(trainSize, batchSize, testSize, nLayers, nNodes, nEpochs, a=0.9, b=1.6):
    
    # function to approximate
    function = lambda s : 1.0/s**12 - 1.0/s**6
    
    # approximate on [a,b]
    a = 0.9
    b = 1.6

    regress = Regression(function, int(1e6), int(1e4), int(1e3))
    regress.generateData(a, b)
    
    # test different activations
    activations = [tf.nn.relu, tf.nn.relu6, tf.nn.elu, tf.nn.sigmoid, tf.nn.tanh]
    counter = 0
    for act in activations:    
        regress.constructNetwork(nLayers, nNodes, activation=act, wInit='trunc_normal', bInit='trunc_normal')
        regress.train(nEpochs, plot=False, counter=counter)
        counter += 1

 
def LennardJonesExample(trainSize, batchSize, testSize, nLayers, nNodes, nEpochs, a=0.9, b=1.6):
    
    function = lambda s : 1.0/s**12 - 1.0/s**6
    regress = Regression(function, trainSize, batchSize, testSize)
    regress.generateData(a, b)
    regress.constructNetwork(nLayers, nNodes, activation=tf.nn.relu, wInit='normal', bInit='normal')
    regress.train(nEpochs, plot=False)
    
    
LennardJonesExample(int(1e6), int(1e4), int(1e3), 2, 4, 50000)
#testActivations(int(1e6), int(1e4), int(1e3), 3, 5, 100000)

    

                                               







