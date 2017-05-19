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
import os
import neuralNetwork as nn
import DataGeneration.lammpsData as lammps
import DataGeneration.symmetries as symmetries
import DataGeneration.readers as readers


# find latest checkpoint
loadDir = "TrainingData/" + sys.argv[1]
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

# set matplotlib parameters
plt.rc('lines', linewidth=1.5)
#plt.rc('axes', prop_cycle=(cycler('color', ['g', 'k', 'y', 'b', 'r', 'c', 'm']) ))
plt.rc('xtick', labelsize=20)
plt.rc('ytick', labelsize=20)
plt.rc('axes', labelsize=25)

class Prettyfloat(float):
    def __repr__(self):
        return "%0.10f" % self
        
    
class Analyze:
    
    def __init__(self, \
                 energy = False, \
                 forces = False, \
                 configSpace = False, \
                 symmetry = False, \
                 numberOfAtoms = 3, \
                 klargerj = True, \
                 cut = False, \
                 tags = False, \
                 forceFile = 'forcesklargerjplus.txt', \
                 plotEnergy=True, \
                 plotForces=True, \
                 plotConfigSpace=True):
        
        self.energy             = energy
        self.forces             = forces
        self.configSpace        = configSpace
        self.symmetry           = symmetry
        self.numberOfAtoms      = numberOfAtoms
        self.klargerj           = klargerj
        self.cut                = cut
        self.tags               = tags
        self.forceFile          = forceFile
        self.plotEnergy         = plotEnergy
        self.plotForces         = plotForces
        self.plotConfigSpace    = plotConfigSpace
        
        self.startSession()
        
        
    def startSession(self):
        
        # read meta file
        metaFile = loadDir + '/meta.dat'
        nNodes, nLayers, activation, self.inputs, outputs, self.lammpsDir = readers.readMetaFile(metaFile)
        
        # read parameters
        self.parameters = readers.readParameters(loadDir + "/parameters.dat")
        self.numberOfSymmFunc = len(self.parameters)
        print "Number of symmetry functions: ", self.numberOfSymmFunc
        
        # begin session
        with tf.Session() as sess:
                
            # construct random NN of same architecture as the one I load
            with tf.name_scope('input'):
               self.x = tf.placeholder('float', [None, self.inputs],  name='x-input')
               self.y = tf.placeholder('float', [None, outputs], name='y-input')
    
            neuralNetwork = nn.neuralNetwork(nNodes, nLayers, activation,
                                             inputs=self.inputs, outputs=outputs)
            makeNetwork = lambda data : neuralNetwork.model(self.x)
            self.prediction = makeNetwork(self.x)
            
            # restore weights and biases from file
            saver = tf.train.Saver()
            saver.restore(sess, loadFileName)
            print 
            print 'Model restored: ', loadFileName 
            
            # read symmetry input values to reconstruct training set
            if self.klargerj:
                if self.cut:
                    symmetryFileName = self.lammpsDir + 'symmetryBehlerklargerjcut.txt'
                else:
                    symmetryFileName = self.lammpsDir + '/symmetryBehlerklargerj.txt'
                print 'Using k > j symmetry vectors'
            else:
                if self.cut:
                    symmetryFileName = self.lammpsDir + '/symmetryBehlerkunequaljcut.txt'
                else:
                    symmetryFileName = self.lammpsDir + '/symmetryBehlerkunequalj.txt'
                print 'Using k != j symmetry vectors'
            
            if os.path.isfile(symmetryFileName):
                print "Reading symmetrized Behler data"
                self.inputData = readers.readSymmetryData(symmetryFileName)
            else: 
                print "Symmetry values file does not exist, has to be made"
                exit(1)
             
            print "Min symm. value: ", np.min(self.inputData)
            print "Max symm. value: ", np.max(self.inputData)
                
            self.numberOfSamples = len(self.inputData)
            self.numberOfTimeSteps = self.numberOfSamples/self.numberOfAtoms
            print "Number of samples: ", self.numberOfSamples
            print "Number of time steps: ", self.numberOfTimeSteps
    
            # read energy forces
            neighbourLists = self.lammpsDir + "/neighbours.txt"
            if self.tags:
                print "Reading forces and tags"
                x0, y0, z0, r0, E, Fx, Fy, Fz, tags = readers.readNeighbourDataForceTag(neighbourLists)
            else:
                print "Reading forces without tags"
                x0, y0, z0, r0, E, Fx, Fy, Fz = readers.readNeighbourDataForce(neighbourLists)
             
            self.x0 = x0
            self.y0 = y0
            self.z0 = z0
            self.r0 = r0
            self.E = np.array(E)
            self.Fx = np.array(Fx)
            self.Fy = np.array(Fy)
            self.Fz = np.array(Fz)

            print "Number of atoms: ", self.numberOfAtoms
            print "Forces is supplied"
              
            # define cost functions
            with tf.name_scope('L2Norm'):
                self.cost = tf.div( tf.nn.l2_loss( tf.subtract(self.prediction, self.y) ), self.numberOfSamples, name='/trainCost')
                tf.summary.scalar('L2Norm', self.cost/self.numberOfSamples)
                
            with tf.name_scope('MAD'):
                MAD = tf.reduce_sum( tf.abs( tf.subtract(self.prediction, self.y) ) )
              
            with tf.name_scope('networkGradient'):
                self.networkGradient = tf.gradients(neuralNetwork.allActivations[-1], self.x)
                
            with tf.name_scope('L2Force'):
                CFDATrain = tf.nn.l2_loss( tf.subtract(self.networkGradient, self.inputData) ) 
            
            if self.energy:
                self.analyzeEnergy(sess)
                
            if self.forces:
                self.analyzeForces(sess)
                
            if self.configSpace:
                self.analyzeConfigSpace(sess)
                
            if self.symmetry:
                self.analyzeSymmetry(sess)
            
            
            
    def analyzeEnergy(self, sess):
        
        x               = self.x
        y               = self.y
        inputData       = self.inputData
        E               = self.E
        numberOfSamples = self.numberOfSamples
        prediction      = self.prediction
        cost            = self.cost
        inputs          = self.inputs
    
        # calculate RMSE of this NN
        RMSE = np.sqrt( 2*sess.run(cost, feed_dict={x: inputData, y: E}) )
        print "RMSE/atom: ", RMSE
        
        # make energy error vs time step plot
        energyNN = np.zeros(numberOfSamples)
        energySW = E[:,0].reshape([numberOfSamples])
        for i in xrange(numberOfSamples):
            energyNN[i] = sess.run(prediction, feed_dict={x: inputData[i].reshape([1,inputs])})
            
        energyError = energyNN - energySW
        aveError = np.sum(energyError) / len(energyError)
        print "Average error: ", aveError
        print "First 5 energies: ", energyNN[:20]
        
        if self.plotEnergy:
            plt.subplot(2,1,1)
            plt.plot(energyNN, 'b-', energySW, 'g-')
            plt.legend([r'$E_{NN}$', r'$E_{SW}$'])
            plt.subplot(2,1,2)
            plt.plot(energyError)
            plt.xlabel('Timestep')
            plt.legend(r'$E_{NN} - E_{SW}$')      
            plt.show() 
    
        
        
    def analyzeForces(self, sess):
        
        x                   = self.x
        networkGradient     = self.networkGradient 
        numberOfSamples     = self.numberOfSamples
        numberOfTimeSteps   = self.numberOfTimeSteps
        numberOfAtoms       = self.numberOfAtoms
        inputs              = self.inputs         
        inputData           = self.inputData
        x0                  = self.x0
        y0                  = self.y0
        z0                  = self.z0
        r0                  = self.r0
        Fx                  = self.Fx
        Fy                  = self.Fy
        Fz                  = self.Fz
                   
        # calculate/read forces for each time step
        forceFile = loadDir + '/' + self.forceFile
        print "Force file: ", forceFile
        if not os.path.isfile(forceFile):
            # calculate forces if not done already
            # calculate deriviative of NN
            dEdG = sess.run(networkGradient, feed_dict={x: inputData})
            dEdG = np.array(dEdG).reshape([numberOfSamples, inputs])
            symmetries.calculateForces(x0, y0, z0, r0, parameters, forceFile, dEdG)
            print
            print "Forces are written to file"

        print "Reading forces"
        if numberOfAtoms == 2:
            forcesNN = np.loadtxt(forceFile)
            FxNN = -forcesNN[:,0].reshape([numberOfSamples,1])
            FyNN = -forcesNN[:,1].reshape([numberOfSamples,1])
            FzNN = -forcesNN[:,2].reshape([numberOfSamples,1])
        else:
            FxNN, FyNN, FzNN = self.readForces(forceFile)
            
        Fx = Fx[np.arange(0,numberOfSamples,numberOfAtoms)].reshape(numberOfTimeSteps)
        Fy = Fy[np.arange(0,numberOfSamples,numberOfAtoms)].reshape(numberOfTimeSteps)
        Fz = Fz[np.arange(0,numberOfSamples,numberOfAtoms)].reshape(numberOfTimeSteps)
        
        Fsw = np.sqrt(Fx**2 + Fy**2 + Fz**2)
        Fnn = np.sqrt(FxNN**2 + FyNN**2 + Fz**2)
        
        xError = FxNN - Fx
        yError = FyNN - Fy
        zError = FzNN - Fz
        absError = Fnn - Fsw 
        
        # compute standard deviation of error
        xStd = np.std(xError)
        yStd = np.std(yError)
        zStd = np.std(zError)
        absStd = np.std(absError)
        print "Std. x: ", xStd
        print "Std. y: ", yStd
        print "Std. z: ", zStd
        print "Std. abs: ", absStd
        
        print 'x0NN - x0SW:', FxNN[:10]/Fx[:10]
        print 'y0NN - y0SW:', FyNN[:10]/Fy[:10]
        
        if self.plotForces:
        
            plt.figure()
            
            plt.subplot(4,2,1)
            plt.plot(FxNN, 'b-', Fx, 'g-')
            plt.legend([r'$F_x^{NN}$', r'$F_x^{SW}$'])
            
            plt.subplot(4,2,3)
            plt.plot(FyNN, 'b-', Fy, 'g-')
            plt.legend([r'$F_y^{NN}$', r'$F_y^{SW}$'])   
            
            plt.subplot(4,2,5)        
            plt.plot(FzNN, 'b-', Fz, 'g-')
            plt.legend([r'$F_z^{NN}$', r'$F_z^{SW}$'])
            
            plt.subplot(4,2,7)       
            plt.plot(Fnn, 'b-', Fsw, 'g-')
            plt.xlabel('Timestep')
            plt.legend([r'$|F|^{NN}$', r'$|F|^{SW}$'])
            
            plt.subplot(4,2,2)
            plt.plot(xError)
            plt.ylabel(r'$\Delta F_x$')
            
            plt.subplot(4,2,4)
            plt.plot(yError)
            plt.ylabel(r'$\Delta F_y$')     
            
            plt.subplot(4,2,6)        
            plt.plot(zError)
            plt.ylabel(r'$\Delta F_z$')
            
            plt.subplot(4,2,8)       
            plt.plot(absError)
            plt.xlabel('Timestep')
            plt.ylabel(r'$\Delta F$')
            
            plt.show()
            
            
    def readForces(self, filename):
        
        numberOfAtoms       = self.numberOfAtoms
        numberOfTimeSteps   = self.numberOfTimeSteps
        
        with open(filename, 'r') as infile:
            
            timeStep = 0
            atom = 1
            FxNN = np.zeros(numberOfTimeSteps)
            FyNN = np.zeros(numberOfTimeSteps) 
            FzNN = np.zeros(numberOfTimeSteps)
            for line in infile:
                words = line.split()
                
                if len(words) != 6:
                    continue
                
                if atom == 1:
                    atom += 1
                    continue
                else:
                    FxNN[timeStep] += float(words[0])
                    FyNN[timeStep] += float(words[1])
                    FzNN[timeStep] += float(words[2])
                                
                    if atom == numberOfAtoms:  
                        timeStep += 1
                        atom = 1
                        continue
                    atom += 1
                    
        print map(Prettyfloat, FxNN[:1])
        print map(Prettyfloat, FyNN[:1])
        print map(Prettyfloat, FzNN[:1])
                    
        return FxNN, FyNN, FzNN


          
    def analyzeConfigSpace(self, sess):  

        x0 = self.x0
        y0 = self.y0
        z0 = self.z0
        r0 = self.r0     
        x = self.x
        y = self.y
        cost = self.cost
        inputData = self.inputData
        E = self.E
        numberOfSymmFunc = self.numberOfSymmFunc
            
        # gather all r in sample
        allX = []; allY = []; allZ = []
        allR = []
        for i in xrange(len(r0)):
            for j in xrange(len(r0[i])):
                allX.append(x0[i][j])
                allY.append(y0[i][j])
                allZ.append(z0[i][j])
                allR.append(r0[i][j])
              
        # sort arrays
        allR = np.array(allR)
        allX = np.array(allX); allY = np.array(allY); allZ = np.array(allZ)
        allR = np.sqrt(allR)        
        p = allR.argsort()
        allR = allR[p]   
        allX = allX[p]
        allY = allY[p]
        allZ = allZ[p]
        
        nPoints = len(allR)
        
        Rik = np.zeros(nPoints) + 2.5
        data = np.hstack((allR,Rik))#.reshape([nPoints,2])
        print data.shape
        print data[4000]
        print self.parameters
        exit(1)
        
        # find cost as function of r
        errors = np.zeros(len(allR))
        for i in xrange(len(allR)):
            errors[i] = sess.run(cost, feed_dict={x : inputData[i].reshape([1,numberOfSymmFunc]), \
                                                  y : outputData[i].reshape([1,1])})
        
        if self.plotConfigSpace:
            plt.hist(allR, bins=100)
            plt.show()
            plt.figure()
            plt.hist(allX, bins=100)
            plt.show()
            plt.figure()
            plt.hist(allY, bins=100) 
            plt.show()
            plt.figure()
            plt.hist(allZ, bins=100)
            plt.show()
        
        plt.plot(allR, errors)
        plt.show()
            
        # evaluate error for all r in set
            
    
    def analyzeSymmetry(self, sess):
        
        inputData = self.inputData.flatten()
        N = len(inputData)
        
        print "Average symmetry value: ", np.average(inputData)
        print "Max symmetry value: ", np.max(inputData)
        print "Min symmetry value: ", np.min(inputData)
        print "Fraction of zeros: ", len(np.where(inputData == 0)[0]) / float(N)
        
        plt.hist(inputData, bins=100)
        plt.show()
        
        
        
        
            
        
        
   
        
        

     
  
    
##### main #####

# energy, forces, configSpace, numberOfAtoms, klargerj, tags, forceFile,
# plotEnergy=True, plotForces=True, plotConfigSpace=True
         
Analyze(energy = False, \
        forces = False, \
        configSpace = False, \
        symmetry = True, \
        numberOfAtoms = 3, \
        klargerj = True, \
        cut = False, \
        tags = False, \
        forceFile = '',\
        plotEnergy=True, \
        plotForces=True, \
        plotConfigSpace=True)

    
    
    
    
    
    
    

 