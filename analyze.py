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
import DataGeneration.symmetries as symmetries


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

forces = True


def readMetaFile(filename):
    
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
        
        return nNodes, nLayers, activation, inputs, outputs, lammpsDir
    


def analyze():
    
    # read meta file
    metaFile = loadDir + '/meta.dat'
    nNodes, nLayers, activation, inputs, outputs, lammpsDir = readMetaFile(metaFile)
    
    # begin session
    with tf.Session() as sess:
            
        # construct random NN of same architecture as the one I load
        with tf.name_scope('input'):
           x = tf.placeholder('float', [None, inputs],  name='x-input')
           y = tf.placeholder('float', [None, outputs], name='y-input')

        neuralNetwork = nn.neuralNetwork(nNodes, nLayers, activation,
                                         inputs=inputs, outputs=outputs)
        makeNetwork = lambda data : neuralNetwork.model(x)
        prediction = makeNetwork(x)
        
        # restore weights and biases from file
        saver = tf.train.Saver()
        saver.restore(sess, loadFileName)
        print 
        print 'Model restored: ', loadFileName 
        
        # read symmetry input values to reconstruct training set
        symmetryFileName = lammpsDir + '/symmetryBehler.txt'
        if os.path.isfile(symmetryFileName):
            print "Reading symmetrized Behler data"
            inputData = lammps.readSymmetryData(symmetryFileName)
        else: 
            print "Symmetry values file does not exist, has to be made"
            exit(1)
        numberOfSamples = len(inputData)
        print "Number of samples: ", numberOfSamples
        
        # read energy and eventual forces
        if forces:
            x0, y0, z0, r0, E, Fx, Fy, Fz = lammps.readNeighbourDataForce(lammpsDir + "/neighbours.txt")
            E = np.array(E)
            Fx = np.array(Fx)
            Fy = np.array(Fy)
            Fz = np.array(Fz)
            print "Forces is supplied"
        else:
            E = lammps.readEnergy(lammpsDir + "/neighbours.txt")
            
        # define cost functions
        with tf.name_scope('L2Norm'):
            cost = tf.div( tf.nn.l2_loss( tf.subtract(prediction, y) ), numberOfSamples, name='/trainCost')
            tf.summary.scalar('L2Norm', cost/numberOfSamples)
            
        with tf.name_scope('MAD'):
            MAD = tf.reduce_sum( tf.abs( tf.subtract(prediction, y) ) )
          
        with tf.name_scope('networkGradient'):
            networkGradient = tf.gradients(neuralNetwork.allActivations[-1], x)
            
        with tf.name_scope('L2Force'):
            CFDATrain = tf.nn.l2_loss( tf.subtract(networkGradient, inputData) ) 
        
        # calculate RMSE of this NN
        RMSE = np.sqrt( 2*sess.run(cost, feed_dict={x: inputData, y: E}) )
        print "RMSE/atom: ", RMSE
        
        # make energy error vs time step plot
        energyNN = np.zeros(numberOfSamples)
        energySW = E[:,0]
        for i in xrange(numberOfSamples):
            energyNN[i] = sess.run(prediction, feed_dict={x: inputData[i].reshape([1,inputs])})
            
        energyError = energyNN - energySW
        aveError = np.sum(energyError) / len(energyError)
        print "Average error: ", aveError
        
        """plt.plot(energyNN, 'b-', energySW, 'g-')
        plt.show()
        plt.plot(energyError)
        plt.show()"""        
        
        # calculate deriviative of NN
        tf.global_variables_initializer()
        dEdG = sess.run(networkGradient, feed_dict={x: inputData})
        dEdG = np.array(dEdG).reshape([numberOfSamples, inputs])
        
        """# read derivatives of symmetry functions
        with open(lammpsDir + "/derivatives.txt", 'r') as infile:
            derivatives = np.zeros((numberOfSamples, 3))
            i = 0
            for line in infile:
                words = line.split()
                derivatives[i,0] = float(words[0])
                derivatives[i,1] = float(words[1])
                derivatives[i,2] = float(words[2])
                i += 1"""
                
        parameters = []
        with open(loadDir + "/parameters.dat", 'r') as infile:
            infile.readline()
            for line in infile:
                param = []
                words = line.split()
                for word in words:
                    param.append(float(word))
                parameters.append(param)
                    
        # calculate forces for each time step
        forceFile = lammpsDir + '/forces.txt'
        if os.path.isfile(forceFile):
            # read file
        else:
            # calculate forces
            symmetries.calculateForces(x0, y0, z0, r0, parameters, lammpsDir, dEdG)
                    
        Forces = np.loadtxt(lammpsDir + '/forces.txt')
        FxNN = Forces[:,0].reshape([numberOfSamples,1])
        FyNN = Forces[:,1].reshape([numberOfSamples,1])
        FzNN = Forces[:,2].reshape([numberOfSamples,1])
         
        """plt.plot(FxNN, 'b-', -Fx, 'g-')
        plt.show()"""
        
        Fsw = np.sqrt(Fx**2 + Fy**2 + Fz**2)
        Fnn = np.sqrt(FxNN**2 + FyNN**2 + Fz**2)
        
        xError = np.abs(FxNN + Fx)
        yError = np.abs(FyNN + Fy)
        zError = np.abs(FzNN + Fz)
        absError = Fnn - Fsw      
        
        # set parameters
        plt.rc('lines', linewidth=1.5)
        #plt.rc('axes', prop_cycle=(cycler('color', ['g', 'k', 'y', 'b', 'r', 'c', 'm']) ))
        plt.rc('xtick', labelsize=20)
        plt.rc('ytick', labelsize=20)
        plt.rc('axes', labelsize=25)
        
        plt.figure()
        
        plt.subplot(4,1,1)
        plt.plot(xError)
        plt.ylabel(r'$\Delta F_x$')
        
        plt.subplot(4,1,2)
        plt.plot(yError)
        plt.ylabel(r'$\Delta F_y$')     
        
        plt.subplot(4,1,3)        
        plt.plot(zError)
        plt.ylabel(r'$\Delta F_z$')
        
        plt.subplot(4,1,4)       
        plt.plot(absError)
        plt.xlabel('Timestep')
        plt.ylabel(r'$\Delta F$')
        
        plt.show()
     
     
    
    
    
    
##### main #####
analyze()

 