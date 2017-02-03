"""
Train a neural network to approximate a continous function
Generate/get training data methods:

size = number of training vectors
LJ: Lennard-Jones
SW: Stillinger-Weber

functionData:       random 1-dim input and output data: 
                    input: r = [size,1], output: E = [size,1] (LJ)
neighbourData:      random N-dim input data (N neighbours)
                    and random 1-dim output data
                    output is total energy of N neighbours
                    input: r = [size,N], output: E = [size,1] (LJ)
                    
                    if functionDerivative is specified (not None):
                    random 4N-dim input data (N neighoburs)
                    and random 4-dimloutput data
                    (x, y, z, r) of each neighbour is supplied
                    output is total energy and force (LJ)
                    input: r = [size,4N], output: [size,4] : (Fx, Fy ,Fz, E)
radialSymmetry:     random M-dim input data and random 1-dim output data
                    use radial symmetry functions to transform radial/coordinates to
                    input: [size, M] where M is number of symmetry functions
                    output: [size, 1] (total energy)
angularSymmetry:    same as for radialSymmetry, but using angular symmetry functions
                    to transform random data. SW potential is used to produce
                    output data
lammps:             (x, y, z, r)  of each neighbour and total E is sampled from lammps
                    input data is transformed with angular symmetry functions
                    same sizes of input and output as for radialSymmetry
                    and angularSymmetry
"""
                    
import tensorflow as tf
import numpy as np
import sys
import datetime as time
import os
import shutil
import matplotlib.pyplot as plt
import DataGeneration.generateData as data
import DataGeneration.lammpsData as lammps
import neuralNetwork as nn
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
                          or sys.argv[i] == '--summary':
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
                     symmFuncType='G4', filename=''):

        self.a, self.b = a, b

        if method == 'functionData':
            self.xTrain, self.yTrain, self.xTest, self.yTest = \
                data.functionData(self.function, self.trainSize, self.testSize, a=a, b=b)
                
        elif method == 'neighbourData:':
            if self.functionDerivative:
                neighbours = self.inputs / 4
                print neighbours
                self.xTrain, self.yTrain = \
                    data.energyAndForceCoordinates(self.function, self.functionDerivative, \
                                                   self.trainSize, \
                                                   neighbours, self.outputs, a, b)
                self.xTest, self.yTest = \
                    data.energyAndForceCoordinates(self.function, self.functionDerivative, \
                                                   self.testSize, \
                                                   neighbours, self.outputs, a, b)

            else:
                self.xTrain, self.yTrain = \
                    data.neighbourData(self.function, self.trainSize, a, b, \
                                       inputs=self.inputs, outputs=self.outputs)
                self.xTest, self.yTest = \
                    data.neighbourData(self.function, self.testSize, a, b, \
                                       inputs=self.inputs, outputs=self.outputs)

        elif method == 'radialSymmetry':
            self.xTrain, self.yTrain = \
                data.radialSymmetryData(self.function, self.trainSize, \
                                        neighbours, numberOfSymmFunc, symmFuncType)
            self.xTest, self.yTest = \
                data.radialSymmetryData(self.function, self.testSize, \
                                        neighbours, numberOfSymmFunc, symmFuncType)

        elif method == 'angularSymmetry':
            self.xTrain, self.yTrain = \
                data.angularSymmetryData(self.function, self.trainSize, \
                                        neighbours, numberOfSymmFunc, symmFuncType, a, b)
            self.xTest, self.yTest = \
                data.angularSymmetryData(self.function, self.testSize, \
                                        neighbours, numberOfSymmFunc, symmFuncType, a, b)

        elif method == 'lammps':
            self.xTrain, self.yTrain, self.xTest, self.yTest, self.inputs, self.outputs = \
                lammps.SiTrainingData(filename, symmFuncType, function=self.function)
            
            # set different sizes based on lammps data
            self.trainSize = self.xTrain.shape[0]
            self.testSize  = self.xTest.shape[0]
            self.batchSize = int(0.8*self.trainSize)
            print "trainSize: ", self.trainSize
            print "testSize: ", self.testSize
            print "batchSize: ", self.batchSize
            print "inputs: ", self.inputs
            print "outputs: ", self.outputs
            


    def constructNetwork(self, nLayers, nNodes, activation=tf.nn.sigmoid, \
                         wInit='normal', bInit='normal', stdDev=1.0):

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
                                              stdDev=stdDev, inputs=self.inputs, outputs=self.outputs)
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
                trainStep = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

            # initialize variables or restore from file
            #saver = tf.train.Saver(self.neuralNetwork.allWeights + self.neuralNetwork.allBiases,
            #                       max_to_keep=None)
            saver = tf.train.Saver()
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
            start = timer()
            for epoch in xrange(numberOfEpochs):

                # pick random batches
                i = np.random.randint(trainSize-batchSize)
                xBatch = xTrain[i:i+batchSize]
                yBatch = yTrain[i:i+batchSize]

                # train
                trainCost, _ = sess.run([cost, trainStep], feed_dict={x: xBatch, y: yBatch})
                
                if summaryFlag:
                    if epoch % 1000 == 0:
                        summary = sess.run(merged, feed_dict={x: xBatch, y: yBatch})
                        train_writer.add_summary(summary, epoch)

                # calculate cost on test set every 1000th epoch
                if epoch % 1000 == 0:
                    testCost = sess.run(cost, feed_dict={x: xTest, y: yTest})
                    print 'Cost/N train test at step %4d: %g %g' % (epoch, trainCost/float(batchSize), \
                                                                    testCost/float(testSize))
                    #sys.stdout.flush()
                    #testCostEnergy = sess.run(costEnergy, feed_dict={x: xTest, y: yTest})
                    #testCostForce = sess.run(costForce, feed_dict={x: xTest, y: yTest})
                    #print 'Cost/N at step %4d: Energy: %g Forces: %g' % \
                    #       (epoch, testCostEnergy/float(testSize), testCostForce/float(testSize))
                    if summaryFlag:
                        summary = sess.run(merged, feed_dict={x: xTest, y: yTest})
                        test_writer.add_summary(summary, epoch)

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
                yy = sess.run(prediction, feed_dict={self.x: x_test})
                plt.plot(x_test[:,0], yy[:,0] - self.function(x_test[:,0]), 'b-')
                #plt.hold('on')
                #plt.plot(x_test[:,0], self.function(x_test[:,0]), 'g-')
                plt.xlabel('r [MD]', fontsize=15)
                plt.ylabel('E [MD]', fontsize=15)
                plt.legend(['NN(r) - LJ(r)'], loc=1)
                plt.savefig(trainingDir + '/errorLJ.pdf', format='pdf')
                #plt.show()


if __name__ == '__main__':

    # trainSize, batchSize, testSize, nLayers, nNodes, nEpochs, ...

    #LennardJonesExample(int(1e6), int(1e4), int(1e3), 2, 4, 100000)
    #testActivations(int(1e6), int(1e4), int(1e3), 3, 5, 100000)
    #LennardJonesNeighbours(int(1e5), int(1e4), int(1e3), 2, 40, int(1e5), 10)
    #LennardJonesNeighboursForce(int(1e5), int(1e4), int(1e3), 2, 100, int(2e6), 5)
    #LennardJonesSymmetryFunctions(int(1e5), int(1e4), int(1e3), 2, 30, int(1e6), 5, 5, '1')
    StillingerWeberSymmetry(int(3e3), int(1e3), int(1e2), 2, 30, int(1e6), 10, 30, 'G3', \
                            "../LAMMPS_test/Silicon/Data/Si1000.xyz")
