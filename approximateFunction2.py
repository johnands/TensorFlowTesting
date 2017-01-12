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
import DataGeneration.generateData as data
import neuralNetworkClass as nn
from Tools.inspect_checkpoint import print_tensors_in_checkpoint_file
from Tools.freeze_graph import freeze_graph
from time import clock as timer

loadFlag            = False
loadFileName        = ''
saveFlag            = False
saveDirName         = ''
summaryFlag         = False
summaryDir          = ''
saveGraphFlag       = False
saveGraphName       = ''
saveGraphProtoFlag  = False
saveGraphProtoName  = ''
saveMetaName        = ''
saveMetaFlag        = False
plotFlag            = False
testFlag            = False

now             = time.datetime.now().strftime("%d.%m-%H.%M.%S")
trainingDir     = 'TrainingData' + '/' + now

# make directory for training data
if len(sys.argv) > 1:
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == '--save' or sys.argv[i] == '--savegraph' or sys.argv[i] == '--savegraphproto' \
           or sys.argv[i] == 'summary':
            if os.path.exists(trainingDir):
                print "Attempted to place data in existing directory, %s. Exiting." % trainingDir
                exit(1)
            else:
                os.mkdir(trainingDir)
                saveMetaName = trainingDir + '/' + 'meta.dat'
                saveGraphName = trainingDir + '/' + 'graph.dat'
                print saveMetaName
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
            saveMetaFlag = True            
            
            # make new directory for checkpoints
            saveDirName 	= trainingDir + '/Checkpoints'
            os.mkdir(saveDirName)

            # Copy the python source code used to run the training, to preserve
            # the tf graph (which is not saved by tf.nn.Saver.save()).
            shutil.copy2(sys.argv[0], saveDirName + '/')
            
        elif sys.argv[i] == '--summary':
            i += 1
            summaryFlag  = True
            saveMetaFlag = True   
            
            # make new directory for summaries
            summaryDir = trainingDir + '/Summaries'
            os.mkdir(summaryDir)
            
        elif sys.argv[i] == '--savegraph':
            i += 1
            saveGraphFlag = True      
            saveMetaFlag = True   
            
        elif sys.argv[i] == '--savegraphproto':
            i += 1
            saveGraphProtoFlag = True
            saveMetaFlag = True   
            
        elif sys.argv[i] == '--plot':
            i += 1
            plotFlag = True
            
        elif sys.argv[i] == '--test':
            i += 1
            testFlag = True
            #numberOfNeighbours = int(sys.argv[i])
            #i += 1
                    
        else:
            i += 1
            
class Regression:
    
    def __init__(self, function, trainSize, batchSize, testSize, inputs, outputs,
                 functionDerivative=None):
        
        self.trainSize = trainSize
        self.batchSize = batchSize
        self.testSize  = testSize
        self.function  = function
        self.inputs    = inputs
        self.outputs   = outputs
        self.functionDerivative = functionDerivative


    def generateData(self, a, b, method='functionData', numberOfSymmFunc=10, neighbours=80, \
                     symmFuncType='1'):
        
        self.a, self.b = a, b
        
        if method == 'functionData':
            self.xTrain, self.yTrain, self.xTest, self.yTest = \
                data.functionData(self.function, self.trainSize, self.testSize, a=a, b=b)
                
        elif method == 'symmetryData':
            self.xTrain, self.yTrain = \
                data.symmetryFunctionsData(self.function, self.trainSize, \
                                           neighbours, numberOfSymmFunc, symmFuncType)
            self.xTest, self.yTest = \
                data.symmetryFunctionsData(self.function, self.testSize, \
                                           neighbours, numberOfSymmFunc, symmFuncType)
                
        else:
            if self.functionDerivative:
                neighbours = self.inputs / 4
                print neighbours
                self.xTrain, self.yTrain = \
                    data.energyAndForeCoordinates(self.function, self.functionDerivative, \
                                                  self.trainSize, \
                                                  neighbours, self.outputs, a, b)
                self.xTest, self.yTest = \
                    data.energyAndForeCoordinates(self.function, self.functionDerivative, \
                                                  self.testSize, \
                                                  neighbours, self.outputs, a, b)
                
            else:        
                self.xTrain, self.yTrain = \
                    data.neighbourData(self.function, self.trainSize, a, b, \
                                       inputs=self.inputs, outputs=self.outputs)
                self.xTest, self.yTest = \
                    data.neighbourData(self.function, self.testSize, a, b, \
                                       inputs=self.inputs, outputs=self.outputs)
        
        
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
                                              stdDev=1.0, inputs=self.inputs, outputs=self.outputs)
        self.makeNetwork = lambda data : self.neuralNetwork.model(self.x)
    
    
    def train(self, numberOfEpochs):    
        
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

            with tf.name_scope('L2Norm'):
                cost = tf.nn.l2_loss( tf.sub(prediction, y) )         
                tf.scalar_summary('L2Norm', cost/batchSize)
            
            #costEnergy = tf.nn.l2_loss(tf.sub(prediction[:,-1], y[:,-1]))
            #costForce = tf.nn.l2_loss(tf.sub(prediction[:,0:-1], y[:,0:-1]))
                      
            with tf.name_scope('train'):
                trainStep = tf.train.AdamOptimizer().minimize(cost)
            
            # initialize variables or restore from file
            #saver = tf.train.Saver(self.neuralNetwork.allWeights + self.neuralNetwork.allBiases, 
            #                       max_to_keep=None)
            saver = tf.train.Saver()
            sess.run(tf.initialize_all_variables())
            if loadFlag:
                saver.restore(sess, loadFileName)
                print 'Model restored'
            
            if testFlag:
                
                # pick an input vector
                coordinates = xTrain[0]
                coordinates = coordinates.reshape([1,self.inputs])
                neighbours = self.inputs/4
                xNN = np.zeros(neighbours)
                yNN = np.zeros(neighbours)
                zNN = np.zeros(neighbours)
                rNN = np.zeros(neighbours)
                # extract coordinates and distances
                for i in range(neighbours):
                    xNN[i] = coordinates[0,i*4]
                    yNN[i] = coordinates[0,i*4 + 1]
                    zNN[i] = coordinates[0,i*4 + 2]
                    rNN[i] = coordinates[0,i*4 + 3]
                                
                # vary coordinates of only one atom and see 
                # if the resulting potential is similar to LJ
                N = 500
                r = np.linspace(0.8, 2.5, N)
                energyNN = []
                energyLJ = []
                forceNN = []
                forceLJ = []
                xyz = np.zeros(3)
                for i in range(N):
                    r2 = r[i]**2
                    xyz[0] = np.random.uniform(0, r2)
                    xyz[1] = np.random.uniform(0, r2-xyz[0])
                    xyz[2] = r2 - xyz[0] - xyz[1]
                    #np.random.shuffle(xyz)
                    x = np.sqrt(xyz[0])# * np.random.choice([-1,1])
                    y = np.sqrt(xyz[1])# * np.random.choice([-1,1])
                    z = np.sqrt(xyz[2])# * np.random.choice([-1,1])                      
                    coordinates[0][0] = x; coordinates[0][1] = y; coordinates[0][2] = z
                    coordinates[0][3] = r[i]
                    energyAndForce = sess.run(prediction, feed_dict={self.x: coordinates})
                    energyNN.append(energyAndForce[0][3]) 
                    rNN[0] = r[i]
                    energyLJ.append(np.sum(self.function(rNN)))
                    forceNN.append(energyAndForce[0][0])
                    xNN[0] = x
                    forceLJ.append(np.sum(self.functionDerivative(rNN)*xNN/rNN))
                
                # convert to arrays
                energyNN = np.array(energyNN); energyLJ = np.array(energyLJ)
                forceNN = np.array(forceNN); forceLJ = np.array(forceLJ)
                
                # plot error 
                plt.plot(r, energyNN - energyLJ)
                plt.xlabel('r [MD]', fontsize=15)
                plt.ylabel('E [MD]', fontsize=15)
                plt.legend(['NN(r) - LJ(r)'], fontsize=15)
                plt.show()
                #plt.savefig('Tests/TrainLennardJones/ManyNeighbourNetwork/Plots/manyNeighbourEnergyError.pdf')
                
                plt.figure()
                plt.plot(r, forceNN - forceLJ)
                plt.xlabel('r [MD]', fontsize=15)
                plt.ylabel('dE/dr [MD]', fontsize=15)
                plt.legend([r'$NN^\prime(r) - LJ^\prime(r)$'], fontsize=15)
                #plt.savefig('Tests/TrainLennardJones/ManyNeighbourNetwork/Plots/manyNeighbourForceError.pdf')
                plt.show()
                #print 'Cost: ', (np.sum((energyNN - energyLJ)**2 + (forceNN - forceLJ)**2))/N
                
                # see if the energy is zero when all neighbours is at cutoff distance
                inputz = np.array([1.87, 1.32, 1.006, 2.5]*neighbours).reshape([1,self.inputs])
                r = np.array([2.5]*neighbours)
                energyLJ = sum(self.function(r))               
                ef = sess.run(prediction, feed_dict={self.x: inputz})
                
                print 'NN energy at cutoff: ', ef[0,3]
                print 'LJ energy at cutoff: ', energyLJ
                
                numberOfEpochs = 0
                
            # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
            if summaryFlag:
                merged = tf.merge_all_summaries()
                train_writer = tf.train.SummaryWriter(summaryDir + '/train', sess.graph)
                test_writer = tf.train.SummaryWriter(summaryDir + '/test')
                    
            # train
            start = timer()
            for epoch in xrange(numberOfEpochs):
                
                # pick random batches
                i = np.random.randint(trainSize-batchSize)
                xBatch = xTrain[i:i+batchSize]
                yBatch = yTrain[i:i+batchSize]
                           
                # calculate cost on test set every 10th epoch
                if epoch % 1000 == 0:               
                    testCost = sess.run(cost, feed_dict={x: xTest, y: yTest})
                    print 'Cost/N at step %4d: %g' % (epoch, testCost/float(testSize))
                    #testCostEnergy = sess.run(costEnergy, feed_dict={x: xTest, y: yTest})
                    #testCostForce = sess.run(costForce, feed_dict={x: xTest, y: yTest})
                    #print 'Cost/N at step %4d: Energy: %g Forces: %g' % (epoch, testCostEnergy/float(testSize), testCostForce/float(testSize))
                    if summaryFlag:
                        summary = sess.run(merged, feed_dict={x: xTest, y: yTest})
                        test_writer.add_summary(summary, epoch)
                
                # train      
                trainCost, _ = sess.run([cost, trainStep], feed_dict={x: xBatch, y: yBatch})
                if summaryFlag:
                    if epoch % 1000 == 0:
                        summary = sess.run(merged, feed_dict={x: xBatch, y: yBatch})
                        train_writer.add_summary(summary, epoch)
                                                        
                # if an argument is passed, save the graph variables ('w', 'b') and dump
                # some info about the training so far to TrainingData/<this run>/meta.dat            
                if saveMetaFlag:
                    if epoch == 0:
                        saveEpochNumber = 0
                        with open(saveMetaName, 'w') as outFile:
                            outStr = '# epochs: %d train: %d, test: %d, batch: %d, nodes: %d, layers: %d \n' \
                                     % (numberOfEpochs, trainSize, testSize, batchSize, nNodes, nLayers)
                            outStr += 'a: %1.1f, b: %1.1f, activation: %s, wInit: %s, bInit: %s' % \
                                       (self.a, self.b, self.activation.__name__, self.wInit, self.bInit)
                            outFile.write(outStr + '\n')
                            outStr = 'Inputs: %d, outputs: %d \n' % (self.inputs, self.outputs)
                            outFile.write(outStr)
                            outStr = '%d %g %g' % \
                                     (epoch, trainCost/float(batchSize), testCost/float(testSize))
                            outFile.write(outStr + '\n')
                    else:
                        if epoch % 1000 == 0:
                             with open(saveMetaName, 'a') as outFile :
                                 outStr = '%d %g %g' % \
                                          (epoch, trainCost/float(batchSize), testCost/float(testSize))
                                 outFile.write(outStr + '\n')                   
                    
                if saveFlag or saveGraphProtoFlag: 
                    if epoch % 1000 == 0:
                        saveFileName = saveDirName + '/' 'ckpt'
                        saver.save(sess, saveFileName, global_step=saveEpochNumber, 
                                   latest_filename="checkpoint_state")
                        saveEpochNumber += 1
                   
            # elapsed time
            end = timer();
            print "Time elapsed: %g" % (end-start)            
       
            # write weights and biases to file when training is finished
            if saveGraphFlag:
                with open(saveGraphName, 'w') as outFile:
                    outStr = "%1d %1d %s %d %d" % (nLayers, nNodes, self.activation.__name__, \
                                                   self.inputs, self.outputs)
                    outFile.write(outStr + '\n')
                    size = len(self.neuralNetwork.allWeights)
                    for i in range(size):
                        weights = sess.run(self.neuralNetwork.allWeights[i])
                        if i < size-1:
                            for j in range(len(weights)):
                                for k in range(len(weights[0])):
                                    outFile.write("%.12g" % weights[j][k])
                                    outFile.write(" ")
                                outFile.write("\n")
                        else:
                            for j in range(len(weights[0])):
                                for k in range(len(weights)):
                                    outFile.write("%.12g" % weights[k][j])
                                    outFile.write(" ")
                                outFile.write("\n")
                            
                    outFile.write("\n")
                        
                    for biasVariable in self.neuralNetwork.allBiases:
                        biases = sess.run(biasVariable)
                        for j in range(len(biases)):
                            outFile.write("%.12g" % biases[j])
                            outFile.write(" ")
                        outFile.write("\n")
            
            # freeze graph
            if saveGraphProtoFlag:
                tf.train.write_graph(sess.graph_def, trainingDir, 'graph.pb')
                
                input_graph_path = trainingDir + '/graph.pb'
                input_saver_def_path = ""
                input_binary = False
                input_checkpoint_path = saveFileName + '-' + str(saveEpochNumber-1)
                output_node_names = "outputLayer/activation"
                restore_op_name = "save/restore_all"
                filename_tensor_name = "save/Const:0"
                output_graph_path = trainingDir + '/frozen_graph.pb'
                clear_devices = False

                freeze_graph(input_graph_path, input_saver_def_path,
                             input_binary, input_checkpoint_path,
                             output_node_names, restore_op_name,
                             filename_tensor_name, output_graph_path,
                             clear_devices, "")
             
            # plot error
            if plotFlag:
                x_test  = np.linspace(self.a, self.b, self.testSize)
                x_test  = x_test.reshape([testSize,self.inputs])
                y_test  = self.function(x_test)
                yy = sess.run(prediction, feed_dict={self.x: x_test})
                plt.plot(x_test[:,0], yy[:,0] - self.function(x_test[:,0]), 'b-')
                #plt.hold('on')
                #plt.plot(x_test[:,0], self.function(x_test[:,0]), 'g-')
                plt.xlabel('r [MD]', fontsize=15)
                plt.ylabel('E [MD]', fontsize=15)
                plt.legend(['NN(r) - LJ(r)'], loc=1)
                plt.savefig(trainingDir + '/errorLJ.pdf', format='pdf')
                #plt.show()

        

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

    regress = Regression(function, int(1e6), int(1e4), int(1e3), 1, 1)
    regress.generateData(a, b)
    
    # test different activations
    activations = [tf.nn.relu, tf.nn.relu6, tf.nn.elu, tf.nn.sigmoid, tf.nn.tanh]
    counter = 0
    for act in activations:    
        regress.constructNetwork(nLayers, nNodes, activation=act, wInit='trunc_normal', bInit='trunc_normal')
        regress.train(nEpochs)
        counter += 1

 
def LennardJonesExample(trainSize, batchSize, testSize, nLayers, nNodes, nEpochs, a=0.8, b=2.5):
    """
    Train to reproduce shifted L-J potential to 
    verify implementation of network and backpropagation in the MD code
    This is a 1-dimensional example
    """
    
    cutoff = 2.5
    shiftedPotential = 1.0/cutoff**12 - 1.0/cutoff**6
    function = lambda s : 4*(1.0/s**12 - 1.0/s**6 - shiftedPotential)
    regress = Regression(function, trainSize, batchSize, testSize, 1, 1)
    regress.generateData(a, b)
    regress.constructNetwork(nLayers, nNodes, activation=tf.nn.sigmoid, \
                             wInit='normal', bInit='normal')
    regress.train(nEpochs)

    
def LennardJonesNeighbours(trainSize, batchSize, testSize, nLayers, nNodes, nEpochs, \
                           neighbours, outputs=1, a=0.8, b=2.5):
                               
    cutoff = 2.5
    shiftedPotential = 1.0/cutoff**12 - 1.0/cutoff**6
    function = lambda s : 4*(1.0/s**12 - 1.0/s**6 - shiftedPotential)
    regress = Regression(function, trainSize, batchSize, testSize, neighbours, outputs)
    regress.generateData(a, b, method='neighbourData')
    regress.constructNetwork(nLayers, nNodes, activation=tf.nn.sigmoid, \
                             wInit='normal', bInit='normal')
    regress.train(nEpochs)
                               
                               
def LennardJonesNeighboursForce(trainSize, batchSize, testSize, nLayers, nNodes, nEpochs, \
                                neighbours, outputs=4, a=0.8, b=2.5):
    
    cutoff = 2.5
    shiftedPotential = 1.0/cutoff**12 - 1.0/cutoff**6
    function = lambda s : 1.0/s**12 - 1.0/s**6 - shiftedPotential
    functionDerivative = lambda t : 12.0/t**13 - 6.0/t**7
    inputs = neighbours*4
    regress = Regression(function, trainSize, batchSize, testSize, inputs, outputs, \
                         functionDerivative)
    regress.generateData(a, b, method='neighbourData')
    regress.constructNetwork(nLayers, nNodes, activation=tf.nn.sigmoid, \
                             wInit='normal', bInit='normal')
    regress.train(nEpochs)
    

def LennardJonesSymmetryFunctions(trainSize, batchSize, testSize, nLayers, nNodes, nEpochs, \
                                  neighbours, numberOfSymmFunc, symmFuncType, outputs=1, a=0.8, b=2.5):
    
    cutoff = 2.5
    shiftedPotential = 1.0/cutoff**12 - 1.0/cutoff**6
    function = lambda s : 1.0/s**12 - 1.0/s**6 - shiftedPotential
    regress = Regression(function, trainSize, batchSize, testSize, numberOfSymmFunc, outputs)
    regress.generateData(a, b, method='symmetryData', neighbours=neighbours, numberOfSymmFunc=numberOfSymmFunc, 
                         symmFuncType='2')
    regress.constructNetwork(nLayers, nNodes, activation=tf.nn.sigmoid, \
                             wInit='normal', bInit='normal')
    regress.train(nEpochs)

# trainSize, batchSize, testSize, nLayers, nNodes, nEpochs, ...

#LennardJonesExample(int(1e6), int(1e4), int(1e3), 2, 4, 100000)
#testActivations(int(1e6), int(1e4), int(1e3), 3, 5, 100000)
#LennardJonesNeighbours(int(1e5), int(1e4), int(1e3), 2, 40, int(1e5), 10)
#LennardJonesNeighboursForce(int(1e5), int(1e4), int(1e3), 2, 100, int(2e6), 5)
LennardJonesSymmetryFunctions(int(1e5), int(1e4), int(1e3), 2, 40, int(1e5), 10, 5, '1')

    

                                               







