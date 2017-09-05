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
import DataGeneration.randomData as data
import DataGeneration.lammpsData as lammps
import DataGeneration.symmetries as symmetries
import neuralNetwork as nn
from Tools.inspect_checkpoint import print_tensors_in_checkpoint_file
from Tools.freeze_graph import freeze_graph
from time import clock as timer
import Tools.matplotlibParameters


loadFlag            = False
loadDir             = ''
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
saveParametersFlag  = False
plotFlag            = False
plotErrorFlag       = False

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
                print "Making data directory: ", saveMetaName
                break
        i += 1


# process command-line input
if len(sys.argv) > 1:
    i = 1
    while i < len(sys.argv):

        if sys.argv[i] == '--load':
            i += 1
            loadFlag     = True
            loadDir = sys.argv[i]
            i += 1
            
            # find latest checkpoint
            loadDir = 'TrainingData/' + loadDir
            checkpointFile = loadDir + "/Checkpoints/checkpoint_state"
            with open(checkpointFile, 'r') as infile:
                line = infile.readline()
                words = line.split()
                checkpoint = words[1][1:-1]
                loadFileName = loadDir + "/Checkpoints/" + checkpoint
                
            print
            print "Information on trained graph: "
            command = "cat " + loadDir + "/README.txt"
            os.system(command)
            print 

        elif sys.argv[i] == '--save':
            i += 1
            saveFlag = True
            saveMetaFlag = True
            print "Checkpoints will be saved"

            # make new directory for checkpoints
            saveDirName 	= trainingDir + '/Checkpoints'
            os.mkdir(saveDirName)

            # copy the python source code used to run the training, to preserve
            # the tf graph (which is not saved by tf.nn.Saver.save()).
            shutil.copy2(sys.argv[0], saveDirName + '/')

        elif sys.argv[i] == '--summary':
            i += 1
            summaryFlag  = True
            saveMetaFlag = True
            print "Summaries will be saved"

            # make new directory for summaries
            summaryDir = trainingDir + '/Summaries'
            os.mkdir(summaryDir)

        elif sys.argv[i] == '--savegraph':
            i += 1
            saveGraphFlag = True
            saveMetaFlag = True
            print "Graph.txt will be saved"

        elif sys.argv[i] == '--savegraphproto':
            i += 1
            saveGraphProtoFlag = True
            saveMetaFlag = True
            print "Binary graph will be saved"

        elif sys.argv[i] == '--plot':
            i += 1
            plotFlag = True
            
        elif sys.argv[i] == '--ploterror':
            i += 1
            plotErrorFlag = True

        else:
            i += 1
            
# copy readme file
if saveFlag and loadFlag:
    os.system("cp " + loadDir + "/README.txt " + trainingDir)
    

class Regression:

    def __init__(self, function, trainSize, batchSize, testSize, inputs, outputs,
                 functionDerivative=False, learningRate=0.001, RMSEtol=1e-10):

        self.trainSize = trainSize
        self.batchSize = batchSize
        self.testSize  = testSize
        self.function  = function
        self.inputs    = inputs
        self.outputs   = outputs
        self.functionDerivative = functionDerivative
        self.learningRate = learningRate
        self.RMSEtol = RMSEtol
        
        # save output to terminal
        if saveFlag or saveGraphFlag:
            filepath = trainingDir + '/output.txt'
            self.outputFile = open(filepath, 'w')
            sys.stdout = self.outputFile


    def generateData(self, a, b, method, numberOfSymmFunc=10, neighbours=80, \
                     symmFuncType='G4', dataFolder='', batch=5, 
                     varyingNeigh=True, forces=False, Behler=True, 
                     klargerj=True, tags=False, atomType=0, nTypes=1, nAtoms=10, 
                     normalize=False, shiftMean=False):

        self.a, self.b = a, b
        self.neighbours = neighbours
        self.forces = forces
        global saveParametersFlag
        self.samplesDir = dataFolder
        self.symmFuncType = symmFuncType

        if method == 'twoBody':
            print "method=twoBody: Generating random, radial 1-neighbour data..."
            self.xTrain, self.yTrain, self.xTest, self.yTest = \
                data.twoBodyEnergy(self.function, self.trainSize, self.testSize, a=a, b=b)
                
            self.numberOfBatches = self.trainSize/self.batchSize
                
        elif method == 'neighbourTwoBody':
            if self.functionDerivative:
                print "method=neighbourTwoBody: Generating random, radial N-neighbour data including force output..."
                neighbours = self.inputs / 4
                print neighbours
                self.xTrain, self.yTrain = \
                    data.neighbourTwoBodyEnergyAndForce2(self.function, self.functionDerivative, \
                                                         self.trainSize, \
                                                         neighbours, self.outputs, a, b)
                self.xTest, self.yTest = \
                    data.neighbourTwoBodyEnergyAndForce2(self.function, self.functionDerivative, \
                                                         self.testSize, \
                                                         neighbours, self.outputs, a, b)

            else:
                print "method=neighbourTwoBody: Generating random, radial N-neighbour data..."
                self.xTrain, self.yTrain = \
                    data.neighbourData(self.function, self.trainSize, a, b, \
                                       inputs=self.inputs, outputs=self.outputs)
                self.xTest, self.yTest = \
                    data.neighbourData(self.function, self.testSize, a, b, \
                                       inputs=self.inputs, outputs=self.outputs)
                                     
        elif method == 'twoBodySymmetry':
            print "method=twoBodySymmetry: Generating random, two-body N-neighbour data with symmetry functions..."
            self.xTrain, self.yTrain, self.parameters = \
                data.neighbourTwoBodySymmetry(self.function, self.trainSize, \
                                              neighbours, numberOfSymmFunc, symmFuncType, a, b,
                                              varyingNeigh=varyingNeigh)
            self.xTest, self.yTest, _ = \
                data.neighbourTwoBodySymmetry(self.function, self.testSize, \
                                              neighbours, numberOfSymmFunc, symmFuncType, a, b,
                                              varyingNeigh=varyingNeigh)
            if saveMetaFlag:
                saveParametersFlag = True

        elif method == 'threeBodySymmetry':
            print "method=threeBodySymmetry: Generating random, three-body N-neighbour data with symmetry functions..."
            self.xTrain, self.yTrain, self.parameters = \
                data.neighbourThreeBodySymmetry(self.function, self.trainSize, \
                                                neighbours, numberOfSymmFunc, symmFuncType, a, b,
                                                varyingNeigh=varyingNeigh)
            self.xTest, self.yTest, _ = \
                data.neighbourThreeBodySymmetry(self.function, self.testSize, \
                                                neighbours, numberOfSymmFunc, symmFuncType, a, b,
                                                varyingNeigh=varyingNeigh)
            self.inputs = len(self.parameters)

            if saveMetaFlag:              
                saveParametersFlag = True

        elif method == 'lammpsSi' or method == 'lammpsSiO2':
            if self.function == None:
                print "method=lammps: Reading data from lammps simulations, including energies..."
            else:
                print "method=lammps: Reading data from lammps simulations, not including energies..."
            
            if not dataFolder:
                print "Path to folder where data is stored must be supplied"
                
            self.samplesDir = dataFolder
            
            # write content of README file to terminal
            print
            print "Content of lammps data file: "
            command = "cat " + dataFolder + "README.txt"
            os.system(command)
            print 
                 
            if method == 'lammpsSi':  
                print 'Training Si'
                self.nTypes = nTypes
                self.xTrain, self.yTrain, self.xTest, self.yTest, self.inputs, self.outputs, self.parameters, \
                self.Ftrain, self.Ftest = \
                    lammps.SiTrainingData(dataFolder, symmFuncType, function=self.function, forces=forces, Behler=Behler, 
                                          klargerj=klargerj, tags=tags, normalize=normalize, shiftMean=shiftMean)
            else:
                print 'Training SiO2'
                self.atomType = atomType
                self.nTypes = nTypes
                self.xTrain, self.yTrain, self.xTest, self.yTest, self.inputs, self.outputs, self.parameters, \
                self.elem2param = \
                    lammps.SiO2TrainingData(dataFolder, symmFuncType, atomType, forces=forces, nAtoms=nAtoms)
            
            # set different sizes based on lammps data
            self.trainSize = self.xTrain.shape[0]
            self.testSize  = self.xTest.shape[0]
            
            ############# EDIT EDIT EDIT change set ###############
            self.xTrain = self.xTrain[:300]
            self.trainSize = self.xTrain.shape[0]
            
            if batch == 1:
                print
                print "Doing offline learning"
            elif batch > 1:
                print 
                print "Doing online learning with", batch, "batches"
            else:
                print "Batch has to be 1 or above, exiting"
                exit(1)
            
            # set batch size, ensure that train size is a multiple of batch size
            rest = self.trainSize % batch
            if rest != 0:
                self.trainSize -= rest
                self.xTrain = self.xTrain[:-rest]
                self.yTrain = self.yTrain[:-rest]
                
            self.batchSize = int(self.trainSize/batch)
            self.numberOfBatches = batch
            
            if saveMetaFlag:
                saveParametersFlag = True
            
        else: 
            print "Invalid data generation method chosen. Exiting..."
            exit(1)
            
        # print out sizes
        print 
        print "##### Training parameters #####"
        print "Training set size: ", self.trainSize
        print "Test set size: ", self.testSize
        print "Batch size: ", self.batchSize
        print "Number of batches: ", self.numberOfBatches
        print "Learning rate: ", self.learningRate
            


    def constructNetwork(self, nLayers, nNodes, activation=tf.nn.sigmoid, \
                         wInit='normal', bInit='normal', stdDev=1.0):

        self.nLayers = nLayers
        self.nNodes  = nNodes
        self.activation = activation
        self.wInit = wInit
        self.bInit = bInit
        
        # print out...
        print
        print "##### network parameters #####"
        print "Inputs: ", self.inputs
        print "Outputs: ", self.outputs
        print "Number of layers: ", nLayers
        print "Number of nodes: ", nNodes
        print "Activation function: ", activation.__name__
        print "Weight initialization: ", wInit
        print "Bias initialization: ", bInit
        print "Setting up NN..."

        # input placeholders
        with tf.name_scope('input'):
            self.x = tf.placeholder('float', [None, self.inputs],  name='x-input')
            self.y = tf.placeholder('float', [None, self.outputs], name='y-input')

        self.neuralNetwork = nn.neuralNetwork(nNodes, nLayers, activation,
                                              weightsInit=wInit, biasesInit=bInit,
                                              stdDev=stdDev, inputs=self.inputs, outputs=self.outputs)
        self.makeNetwork = lambda data : self.neuralNetwork.model(self.x)
        


    def train(self, numberOfEpochs):

        trainSize       = self.trainSize
        batchSize       = self.batchSize
        testSize        = self.testSize
        xTrain          = self.xTrain
        yTrain          = self.yTrain
        xTest           = self.xTest
        yTest           = self.yTest
        numberOfBatches = self.numberOfBatches
        x               = self.x
        y               = self.y
        nNodes          = self.nNodes
        nLayers         = self.nLayers
        learningRate    = self.learningRate

        # begin session
        with tf.Session() as sess:

            # pass data to network and receive output
            prediction = self.makeNetwork(x)

            with tf.name_scope('L2Norm'):
                # HAVE CHANGED HERE!!!!
                trainCost = tf.div( tf.nn.l2_loss( tf.subtract(prediction, y) ), batchSize, name='/trainCost')
                testCost  = tf.div( tf.nn.l2_loss( tf.subtract(prediction, y) ), testSize, name='/testCost')
                tf.summary.scalar('L2Norm', trainCost/batchSize)
                
            with tf.name_scope('MAD'):
                MAD = tf.reduce_sum( tf.abs( tf.subtract(prediction, y) ) )

            with tf.name_scope('train'):
                trainStep = tf.train.AdamOptimizer(learning_rate=learningRate).minimize(trainCost)
              
            with tf.name_scope('networkGradient'):
                networkGradient = tf.gradients(self.neuralNetwork.allActivations[-1], x)
                
            #with tf.name_scope('L2Force'):
            #    CFDATrain = tf.nn.l2_loss( tf.subtract(networkGradient, xTrain) )

            # initialize variables or restore from file
            saver = tf.train.Saver(keep_checkpoint_every_n_hours=1)
            sess.run(tf.global_variables_initializer())
            if loadFlag:
                saver.restore(sess, loadFileName)
                print 'Model %s restored' % loadFileName             
            
            # merge all the summaries and write them out to training directory
            if summaryFlag:
                merged = tf.merge_all_summaries()
                train_writer = tf.train.SummaryWriter(summaryDir + '/train', sess.graph)
                test_writer = tf.train.SummaryWriter(summaryDir + '/test')
                
            # decide how often to print and store things
            every = 1000/self.numberOfBatches
            
            # EDIT EDIT EDIT 
            every = 10
            
            if loadFlag and (plotFlag or plotErrorFlag) and not saveFlag:
                numberOfEpochs = -1

            # train
            print 
            print "##### Starting training session #####"
            start = timer()
            for epoch in xrange(numberOfEpochs+1): 
                
                # for shuffling the training set
                indicies = np.random.choice(trainSize, trainSize, replace=False)
                
                # offline learning
                if batchSize == trainSize:    
                    
                    # pick whole set in random order               
                    xBatch = xTrain[indicies]
                    yBatch = yTrain[indicies]
                    
                    # train
                    sess.run(trainStep, feed_dict={x: xBatch, y: yBatch})
                    
                # online learning
                else:                      
                    # loop through whole set, train each iteration
                    for b in xrange(numberOfBatches):
                        batch = indicies[b*batchSize:(b+1)*batchSize]
                        xBatch = xTrain[batch]
                        yBatch = yTrain[batch]
                        
                        # train
                        sess.run(trainStep, feed_dict={x: xBatch, y: yBatch})
                
                if summaryFlag:
                    if not epoch % every:
                        summary = sess.run(merged, feed_dict={x: xBatch, y: yBatch})
                        train_writer.add_summary(summary, epoch)

                # calculate cost every 1000th epoch
                if not epoch % every:
                    trainError, absErrorTrain = sess.run([trainCost, MAD], feed_dict={x: xBatch, y: yBatch})
                    testError, absErrorTest   = sess.run([testCost, MAD], feed_dict={x: xTest, y: yTest})
                    trainRMSE = np.sqrt(2*trainError)
                    testRMSE = np.sqrt(2*testError)
                    print 'Cost/N train test at epoch %4d: TF: %g %g, RMSE: %g %g, MAD: %g %g' % \
                                                    ( epoch, trainError, testError, \
                                                      trainRMSE, \
                                                      testRMSE, \
                                                      absErrorTrain/float(batchSize), \
                                                      absErrorTest/float(testSize) )
                    sys.stdout.flush()

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
                            outStr += 'a: %1.1f, b: %1.1f, activation: %s, wInit: %s, bInit: %s, learnRate: %g, symm: %s' % \
                                       (self.a, self.b, self.activation.__name__, self.wInit, self.bInit, self.learningRate, self.symmFuncType)
                            outFile.write(outStr + '\n')
                            outStr = 'Inputs: %d, outputs: %d, loaded: %s, sampled: %s \n' %  \
                                     (self.inputs, self.outputs, loadDir, self.samplesDir)
                            outFile.write(outStr)
                            outStr = '%d %g %g' % \
                                     (epoch, trainRMSE, testRMSE)
                            outFile.write(outStr + '\n')
                    else:
                        if not epoch % every:
                             with open(saveMetaName, 'a') as outFile :
                                 outStr = '%d %g %g' % \
                                          (epoch, trainRMSE, testRMSE)
                                 outFile.write(outStr + '\n')

                if saveFlag or saveGraphProtoFlag:
                    if not epoch % every:
                        saveFileName = saveDirName + '/' 'ckpt'
                        saver.save(sess, saveFileName, global_step=epoch,
                                   latest_filename="checkpoint_state")
                        
                # finish training if RMSE of test set is below tolerance
                if testRMSE < self.RMSEtol:
                    print "Reached RMSE tolerance"
                    break
            
            if numberOfEpochs == -1:
                print sess.run(trainCost, feed_dict={x: xTrain[0].reshape([1,self.inputs]), y: yTrain[0].reshape([1,1])})
                print xTrain[0]
                print yTrain[0]
                print '%.16g' % sess.run(prediction, feed_dict={x: xTrain[0].reshape([1,self.inputs])})
                print  sess.run(networkGradient, feed_dict={x: xTrain})                                           
                        
            # elapsed time
            end = timer();
            print "Time elapsed: %g" % (end-start)

            # write network to file when training is finished
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
                                    outFile.write("%.16g" % weights[j][k])
                                    outFile.write(" ")
                                outFile.write("\n")
                        else:
                            for j in range(len(weights[0])):
                                for k in range(len(weights)):
                                    outFile.write("%.16g" % weights[k][j])
                                    outFile.write(" ")
                                outFile.write("\n")

                    outFile.write("\n")

                    for biasVariable in self.neuralNetwork.allBiases:
                        biases = sess.run(biasVariable)
                        for j in range(len(biases)):
                            outFile.write("%.16g" % biases[j])
                            outFile.write(" ")
                        outFile.write("\n")

            # save parameters to file
            if saveParametersFlag:
                parameters = self.parameters
                numberOfParameters = len(parameters[0])
                saveParametersName = trainingDir + '/' + 'parameters.dat'
                
                if self.nTypes > 1:
                    with open(saveParametersName, 'w') as outFile:   
                        outStr = "%d %d" % (len(parameters), self.atomType)
                        outFile.write(outStr + '\n')
                        # G2
                        for jtype in xrange(self.nTypes):
                            key = (self.atomType, jtype)
                            if key in self.elem2param:
                                interval = self.elem2param[(self.atomType,jtype)]
                                for s, p in enumerate(parameters[interval[0]:interval[1]], interval[0]):
                                    for param in p:
                                        outFile.write("%g " % param)
                                    outFile.write("\n")
                                outFile.write("\n")
                            
                        # G4/G5
                        for jtype in xrange(self.nTypes):
                            for ktype in xrange(self.nTypes):
                                key = (self.atomType, jtype, ktype)
                                if key in self.elem2param:
                                    interval = self.elem2param[(self.atomType,jtype,ktype)]
                                    for s, p in enumerate(parameters[interval[0]:interval[1]], interval[0]):
                                        for param in p:
                                            outFile.write("%g " % param)
                                        outFile.write("\n")
                                    outFile.write("\n")     
                                    
                else:
                    with open(saveParametersName, 'w') as outFile:
                        # write number of symmfuncs and number of unique parameters
                        outStr = "%d" % len(parameters)
                        outFile.write(outStr + '\n')
                        for symmFunc in range(len(parameters)):
                            for param in parameters[symmFunc]:
                                outFile.write("%g" % param)
                                if numberOfParameters > 1:
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

            if saveFlag or saveGraphFlag:
                self.outputFile.close()
            
            # plot RMSE as function of epoch
            if plotFlag:
                
                if loadFlag and not saveFlag:
                    location = loadDir + '/meta.dat'
                else:
                    location = saveMetaName
                
                with open(location) as infile:
                    
                    # skip headers
                    infile.readline(); infile.readline(); infile.readline()
                    
                    # read RMSE of train and test
                    epoch = []; trainError = []; testError = [];
                    for line in infile:
                        words = line.split()
                        epoch.append(float(words[0]))
                        trainError.append(float(words[1]))
                        testError.append(float(words[2]))
                        
                plt.plot(epoch, trainError, 'b-', epoch, testError, 'g-')
                plt.xlabel('Epoch')
                plt.ylabel('RMSE')
                plt.legend(['Training set', 'Test set'], prop={'size':20})
                plt.axis([0, 5000, 0, max(trainError + testError)])
                plt.tight_layout()
                #plt.savefig('../Oppgaven/Figures/Implementation/overfitting.pdf')
                plt.show()
                
            # plot error of 1-dim function on interval [a,b]    
            if plotErrorFlag:
                
                # make interval [a,b]
                N = 2000
                interval = np.linspace(self.a, self.b, N)
                
                # evaluate trained network on this interval 
                energiesNN = sess.run(prediction, feed_dict={x: interval.reshape([N,1])})
                energiesLJ = self.function(interval)
                    
                plt.plot(interval, energiesLJ, 'b-', interval, energiesNN, 'g-')
                plt.legend(['LJ', 'NN'])
                plt.show()
                
                energyError = energiesLJ - energiesNN.flatten()
                print "RMSE: ", np.sqrt(np.sum(energyError**2)/N)
                plt.figure()
                plt.plot(interval, energyError)
                plt.xlabel(r'$r \, [\mathrm{\AA{}}]$')
                plt.ylabel(r'$E_{\mathrm{LJ}} - E_{\mathrm{NN}} \, [eV]$')
                plt.legend(['Absolute error'], prop={'size':20})
                plt.tight_layout()
                plt.savefig('../Oppgaven/Figures/Implementation/LJError.pdf')
                #plt.show()
                
                plt.figure()
                plt.plot(interval, np.abs(energyError))
                plt.show()
                    
                
                
                    
                
                



if __name__ == '__main__':
    pass