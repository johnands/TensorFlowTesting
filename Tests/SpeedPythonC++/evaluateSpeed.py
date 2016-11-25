import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
grandParentDir = os.path.dirname(parentdir)
sys.path.insert(0, parentdir) 
sys.path.insert(1, grandParentDir)

import tensorflow as tf
import numpy as np
import DataGeneration.generateData as trainingData
from Tools.freeze_graph import freeze_graph
import neuralNetworkClass as nn
#from timeit import default_timer as timer
from time import clock as timer


##### argument parser #####

saveGraphFlag       = False
saveGraphProtoFlag  = False

if len(sys.argv) > 1:
    i = 1
    while i < len(sys.argv):
            
        if sys.argv[i] == '--savegraph':
            i += 1
            saveGraphFlag = True           
            
        elif sys.argv[i] == '--savegraphproto':
            i += 1
            saveGraphProtoFlag = True
                    
        else:
            i += 1


def constructNetwork(inputs, outputs, nLayers, nNodes, activation=tf.nn.sigmoid, \
                     wInit='normal', bInit='normal'):
                  
    # input placeholders
    with tf.name_scope('input'):
        x = tf.placeholder('float', [None, inputs],  name='x-input')
        y = tf.placeholder('float', [None, outputs], name='y-input')

    
    # set up network
    neuralNetwork = nn.neuralNetwork(nNodes, nLayers, activation,
                                          weightsInit=wInit, biasesInit=bInit,
                                          stdDev=1.0)
    makeNetwork = lambda data : neuralNetwork.model(x)
    
    return x, y, makeNetwork, neuralNetwork
    

# function to approximate
function = lambda s : 1.0/s**12 - 1.0/s**6
trainSize = 100; testSize = 100
a = 0.8; b = 2.5
N = 50
xTrain, yTrain, xTest, yTest = \
    trainingData.functionData(function, trainSize, testSize, a, b)

if saveGraphFlag:
    with open('evaluateNetworkTest.dat', 'w') as outFile:
        outFile.write('nLayers nNodes Time')
        outFile.write('\n')
        
# compute time for different architectures
maxLayer = 30
maxNodes = 100
for layers in range(1, maxLayer+1, 1):
    for nodes in range(1, maxNodes+1, 1):
        time = 0
        tf.reset_default_graph()
        
        with tf.Session() as sess:
            
            # make and initialize network
            x, y, network, neuralNetwork = constructNetwork(1, 1, layers, nodes) 
            prediction = network(x) 
            sess.run(tf.initialize_all_variables())
            
            # run multiple times with different inputs
            for i in xrange(N): 
                randomInt = np.random.randint(trainSize)
                dataPoint = xTrain[randomInt].reshape([1,1])
                start = timer()
                sess.run(prediction, feed_dict={x: dataPoint})
                end = timer()
                time += end-start 
            
            if saveGraphProtoFlag:
                # save current variables
                saver = tf.train.Saver()
                saveName = 'Checkpoints/ckpt-L%1dN%d' % (layers, nodes)
                saver.save(sess, saveName)
                                   
                # save current graph
                graphName = 'graphL%1dN%d.pb' % (layers, nodes)
                tf.train.write_graph(sess.graph_def, "Graphs", graphName)
                
                # freeze graph
                input_graph_path = 'Graphs/' + graphName
                input_saver_def_path = ""
                input_binary = False
                input_checkpoint_path = saveName
                output_node_names = "outputLayer/activation"
                restore_op_name = "save/restore_all"
                filename_tensor_name = "save/Const:0"
                output_graph_path = 'Graphs/frozen_' + graphName
                clear_devices = False
    
                freeze_graph(input_graph_path, input_saver_def_path,
                             input_binary, input_checkpoint_path,
                             output_node_names, restore_op_name,
                             filename_tensor_name, output_graph_path,
                             clear_devices, "")
                             

            if saveGraphFlag:                
                with open('evaluateNetworkTest.dat', 'a') as outFile:
                    outFile.write('%2d %2d %g' % (layers, nodes, time/float(N)))
                    outFile.write('\n')
                    
                # save graph manually      
                graphManualName = 'ManualGraphs/graphL%1dN%d.dat' % (layers, nodes)
                with open(graphManualName, 'w') as outFile:
                    outStr = "%1d %1d sigmoid" % (layers, nodes)
                    outFile.write(outStr + '\n')
                    size = len(neuralNetwork.allWeights)
                    for i in range(size):
                        weights = sess.run(neuralNetwork.allWeights[i])
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
                        
                    for biasVariable in neuralNetwork.allBiases:
                        biases = sess.run(biasVariable)
                        for j in range(len(biases)):
                            outFile.write("%.12g" % biases[j])
                            outFile.write(" ")
                        outFile.write("\n")
                
        # show progress
        sys.stdout.write("\r%2d %% complete" % ((float(layers)/maxLayer)*100))
        sys.stdout.flush()
                

                
        

    
    
    
    
        

