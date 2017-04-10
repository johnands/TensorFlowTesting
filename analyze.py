"""
Load a trained graph to analyze the error of the NN 
and its derivatives
1. Plot error as function of time step in MD simulation, i.e.
compare the error on the configs sampled from lammps
2. Plot error as function of a certain distance or angle interval 
(not very easy)
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys
from Tools.inspect_checkpoint import print_tensors_in_checkpoint_file
import os
import neuralNetwork as nn
import DataGeneration.lammpsData as lammps


# find latest checkpoint
loadDir = sys.argv[1]
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


def analyze():
    
    with tf.Session() as sess:
        
        # must first create a NN with the same architecture as the
        # on I want to load
        with open(loadDir + '/meta.dat', 'r') as infile:
            
            # read number of nodes and layers
            words = infile.readline().split()
            nNodes = int(words[-3][:-1])
            nLayers = int(words[-1])
            print "Number of nodes: ", nNodes
            print "Number of layers: ", nLayers
            
            # read activation function
            words = infile.readline().split()
            activation = words[5][:-1]
            print "Activation: ", activation
            if activation == 'sigmoid':
                activation = tf.nn.sigmoid
            elif activation == 'tanh':
                activation = tf.nn.tanh
            else:
                print activation, " is not a valid activation"
                
            # read inputs, outputs and lammps sample folder
            words = infile.readline().split()
            inputs = int(words[1][:-1])
            outputs = int(words[3][:-1])
            lammpsDir = words[-1]
            print "Inputs: ", inputs
            print "Outputs: ", outputs
            print "Lammps folder: ", lammpsDir
            
        # construct NN
        with tf.name_scope('input'):
           x = tf.placeholder('float', [None, inputs],  name='x-input')
           y = tf.placeholder('float', [None, outputs], name='y-input')

        neuralNetwork = nn.neuralNetwork(nNodes, nLayers, activation,
                                         inputs=inputs, outputs=outputs)
        makeNetwork = lambda data : neuralNetwork.model(x)
        
        # read input and output data (if file do not exist, apply symmetry)
        # comment out this line if data from 10/4 -->
        #lammpsDir = '../LAMMPS_test/Silicon/Data/' + lammpsDir

        symmetryFileName = lammpsDir + '/symmetryBehler.txt'
        if os.path.isfile(symmetryFileName):
            print "Reading symmetrized Behler data"
            inputData, outputData = lammps.readSymmetryData(symmetryFileName)
        else: 
            print "Symmetry values file does not exist, has to be made"
            exit(1)
        numberOfSamples = len(inputData)
        print "Number of samples: ", numberOfSamples
        
        # pass data to network and receive output
        prediction = makeNetwork(x)

        with tf.name_scope('L2Norm'):
            cost = tf.div( tf.nn.l2_loss( tf.subtract(prediction, y) ), numberOfSamples, name='/trainCost')
            tf.summary.scalar('L2Norm', cost/numberOfSamples)
            
        with tf.name_scope('MAD'):
            MAD = tf.reduce_sum( tf.abs( tf.subtract(prediction, y) ) )
          
        with tf.name_scope('networkGradient'):
            networkGradient = tf.gradients(neuralNetwork.allActivations[-1], x)
            
        with tf.name_scope('L2Force'):
            CFDATrain = tf.nn.l2_loss( tf.subtract(networkGradient, inputData) )

        # initialize variables or restore from file
        saver = tf.train.Saver()
        saver.restore(sess, loadFileName)
        print 
        print 'Model restored: ', loadFileName  
        
        RMSE = np.sqrt( 2*sess.run(cost, feed_dict={x: inputData, y: outputData}) )
        print "RMSE/atom: ", RMSE
        
        # make energy error vs time step plot
        energyNN = np.zeros(numberOfSamples)
        energySW = outputData[:,0]
        for i in xrange(numberOfSamples):
            energyNN[i] = sess.run(prediction, feed_dict={x: inputData[i].reshape([1,inputs])})
            
        energyError = energyNN - energySW
        aveError = np.sum(energyError) / len(energyError)
        print "Average error: ", aveError
        
        plt.plot(energyNN, 'b-', energySW, 'g-')
        plt.show()
        plt.plot(energyError)
        plt.show()
        
        
        
            
            
                
                
        
    

     
     
    
    
    
    
##### main #####
analyze()

 