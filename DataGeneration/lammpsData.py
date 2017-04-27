# read data from LAMMPS runs for training

import numpy as np
import sys
import symmetries
import readers
import os
 
    
def parametersBehler():
    
    # make nested list of all symmetry function parameters
    # parameters from Behler
    parameters = []    
    
    # type1
    center = 0.0
    cutoff = 6.0
    for eta in [2.0, 0.5, 0.2, 0.1, 0.04, 0.001]:
        parameters.append([eta, cutoff, center])
    
    # type2
    zeta = 1.0
    inversion = 1.0
    eta = 0.01
    for cutoff in [6.0, 5.5, 5.0, 4.5, 4.0, 3.5]:
        parameters.append([eta, cutoff, zeta, inversion])
        
    # type 3
    cutoff = 6.0
    eta = 4.0
    for center in [5.5, 5.0, 4.5, 4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0]:
        parameters.append([eta, cutoff, center])
        
        
    eta = 0.01
    
    # type 4
    zeta = 1.0
    inversion = -1.0    
    for cutoff in [6.0, 5.5, 5.0, 4.5, 4.0, 3.5]:
        parameters.append([eta, cutoff, zeta, inversion])
        
    # type 5 and 6
    zeta = 2.0
    for inversion in [1.0, -1.0]:
        for cutoff in [6.0, 5.0, 4.0, 3.0]:
            parameters.append([eta, cutoff, zeta, inversion])
        
    # type 7 and 8
    zeta = 4.0
    for inversion in [1.0, -1.0]:
        for cutoff in [6.0, 5.0, 4.0, 3.0]:
            parameters.append([eta, cutoff, zeta, inversion])
    
    # type 9 and 10
    zeta = 16.0
    for inversion in [1.0, -1.0]:
        for cutoff in [6.0, 4.0]:
            parameters.append([eta, cutoff, zeta, inversion])  
            
    return parameters
    

def parametersCustomized():
    
    # make nested list of all symmetry function parameters
    # parameters from Behler
    parameters = []    
    
    # type1
    center = 0.0
    cutoff = 3.77118
    for eta in [2.0, 0.5, 0.2, 0.1, 0.04, 0.001]:
        parameters.append([eta, cutoff, center])
    
    # type2
    zeta = 1.0
    inversion = 1.0
    eta = 0.01
    for cutoff in [6.0, 5.5, 5.0, 4.5, 4.0, 3.8]:
        parameters.append([eta, cutoff, zeta, inversion])
        
    # type 3
    cutoff = 6.0
    eta = 4.0
    for center in [5.5, 5.0, 4.5, 4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0]:
        parameters.append([eta, cutoff, center])
        
        
    eta = 0.01
    
    # type 4
    zeta = 1.0
    inversion = -1.0    
    for cutoff in [6.0, 5.5, 5.0, 4.5, 4.0, 3.8]:
        parameters.append([eta, cutoff, zeta, inversion])
        
    # type 5 and 6
    zeta = 2.0
    for inversion in [1.0, -1.0]:
        for cutoff in [6.0, 5.0, 4.0, 3.8]:
            parameters.append([eta, cutoff, zeta, inversion])
        
    # type 7 and 8
    zeta = 4.0
    for inversion in [1.0, -1.0]:
        for cutoff in [6.0, 5.0, 4.0, 3.8]:
            parameters.append([eta, cutoff, zeta, inversion])
    
    # type 9 and 10
    zeta = 16.0
    for inversion in [1.0, -1.0]:
        for cutoff in [6.0, 4.0]:
            parameters.append([eta, cutoff, zeta, inversion])  
            
    return parameters
    


def SiTrainingData(filename, symmFuncType, function=None, forces=False, Behler=True, 
                   klargerj=False, tags=True):
    """ 
    Coordinates and energies of neighbours is sampled from lammps
    Angular symmtry funcitons are used to transform input data  
    """
     
    # read file
    if forces:
        if tags:
            print 
            print "Tags are included in neighbour lists"
            x, y, z, r, E, Fx, Fy, Fz, _ = readers.readNeighbourDataForceTag(filename)
        else:
            print 
            print "Tags are not included in neighbour lists"
            x, y, z, r, E, Fx, Fy, Fz = readers.readNeighbourDataForce(filename)
        Fx = np.array(Fx)
        Fy = np.array(Fy)
        Fz = np.array(Fz)
        print "Forces is applied"
    else:
        x, y, z, r, E = readers.readNeighbourData(filename)
    print "Neighbour list file is read..."

    # make nested list of all symmetry function parameters
    # parameters from Behler
    if Behler:
        print 
        print "Using Behler parameters"
        parameters = parametersBehler()
    else:
        parameters = parametersCustomized()
        print
        print "Using customized parameters" 
                    
    numberOfSymmFunc = len(parameters)
    outputs = 1
    
    # decide on which symmetry parameters set to use
    sampleDir = filename[:-14]
    if Behler:
        if klargerj:
            print "k > j"
            symmetryFileName = sampleDir + 'symmetryBehlerklargerjcut.txt'
        else:
            print "k != j"
            symmetryFileName = sampleDir + 'symmetryBehlerkunequaljcut.txt'
            
    else:
        if klargerj:
            print "k > j"
            symmetryFileName = sampleDir + 'symmetryCustomklargerjG4.txt'
        else:
            print "k != j"
            symmetryFileName = sampleDir + 'symmetryCustomkunequalj.txt'
        print "Using customized symmetry parameters"
            
    # apply symmetry or read already existing file
    if os.path.isfile(symmetryFileName):
        print "Reading symmetrized input data"
        inputData = readers.readSymmetryData(symmetryFileName)
        outputData = np.array(E)
        print "Energy is supplied from lammps"
    else: 
        # apply symmetry transformastion
        inputData, outputData = symmetries.applyThreeBodySymmetry(x, y, z, r, parameters, symmFuncType, \
                                                                  function=function, E=E, sampleName=symmetryFileName, 
                                                                  forces=forces, klargerj=klargerj)
        print 'Applying symmetry transformation'
        
        
    # split in training set and test set randomly
    totalSize       = len(inputData)
    if len(inputData) < 10:
        testSize = 1
    else:
        testSize        = int(0.1*totalSize) 
    indicies        = np.random.choice(totalSize, testSize, replace=False)
    inputTest       = inputData[indicies]         
    outputTest      = outputData[indicies]
    inputTraining   = np.delete(inputData, indicies, axis=0)
    outputTraining  = np.delete(outputData, indicies, axis=0)
    
    """with open(sampleDir + 'symmetryBehlerTrain.txt', 'w') as outfile:
        for vector in inputTraining:
            for symmValue in vector:
                outfile.write('%g ' % symmValue)
            outfile.write('\n')
        outfile.write('\n')
        for vector in outputTraining:
            for symmValue in vector:
                outfile.write('%g ' % symmValue)
                outfile.write('\n')
                
    with open(sampleDir + 'symmetryBehlerTest.txt', 'w') as outfile:
        for vector in inputTest:
            for symmValue in vector:
                outfile.write('%g ' % symmValue)
            outfile.write('\n')
        outfile.write('\n')
        for vector in outputTest:
            for symmValue in vector:
                outfile.write('%g ' % symmValue)
                outfile.write('\n')"""
    
    if forces:
        FxTest = Fx[indicies]
        FyTest = Fy[indicies]
        FzTest = Fz[indicies]
        FxTrain = np.delete(Fx, indicies, axis=0)
        FyTrain = np.delete(Fy, indicies, axis=0)
        FzTrain = np.delete(Fz, indicies, axis=0)
        
        Ftest = []; Ftrain = []
        Ftest.append(FxTest)
        Ftest.append(FyTest)
        Ftest.append(FzTest)
        Ftrain.append(FxTrain)
        Ftrain.append(FyTrain)
        Ftrain.append(FzTrain) 
    else:
        Ftest = None
        Ftrain = None
   
    return inputTraining, outputTraining, inputTest, outputTest, numberOfSymmFunc, outputs, parameters, Ftrain, Ftest 
    

def SiO2TrainingData(filename, symmFuncType, function=None):
    """ 
    Coordinates and energies of neighbours is sampled from lammps
    Angular symmtry funcitons are used to transform input data  
    """
    
    # read file
    x, y, z, r, E = readers.readNeighbourData(filename)
    print "File is read..."

    # make nested list of all symetry function parameters
    # parameters from Behler
    parameters = []    
    
    # type1
    center = 0.0
    cutoff = 6.0
    for eta in [2.0, 0.5, 0.2, 0.1, 0.04, 0.001]:
        parameters.append([eta, cutoff, center])
    
    # type2
    zeta = 1.0
    inversion = 1.0
    eta = 0.01
    for cutoff in [6.0, 5.5, 5.0, 4.5, 4.0, 3.5]:
        parameters.append([eta, cutoff, zeta, inversion])
        
    # type 3
    cutoff = 6.0
    eta = 4.0
    for center in [5.5, 5.0, 4.5, 4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0]:
        parameters.append([eta, cutoff, center])
        
        
    eta = 0.01
    
    # type 4
    zeta = 1.0
    inversion = -1.0    
    for cutoff in [6.0, 5.5, 5.0, 4.5, 4.0, 3.5]:
        parameters.append([eta, cutoff, zeta, inversion])
        
    # type 5 and 6
    zeta = 2.0
    for inversion in [1.0, -1.0]:
        for cutoff in [6.0, 5.0, 4.0, 3.0]:
            parameters.append([eta, cutoff, zeta, inversion])
        
    # type 7 and 8
    zeta = 4.0
    for inversion in [1.0, -1.0]:
        for cutoff in [6.0, 5.0, 4.0, 3.0]:
            parameters.append([eta, cutoff, zeta, inversion])
    
    # type 9 and 10
    zeta = 16.0
    for inversion in [1.0, -1.0]:
        for cutoff in [6.0, 4.0]:
            parameters.append([eta, cutoff, zeta, inversion])   
                    
    numberOfSymmFunc = len(parameters)
    outputs = 1
                   
    # apply symmetry transformastion
    inputData, outputData = symmetries.applyThreeBodySymmetry(x, y, z, r, parameters, symmFuncType, function=function, E=E)
    
    # split in training set and test set randomly
    totalSize       = len(inputData)
    testSize        = int(0.1*totalSize) 
    indicies        = np.random.choice(totalSize, testSize, replace=False)
    inputTest       = inputData[indicies]         
    outputTest      = outputData[indicies] 
    inputTraining   = np.delete(inputData, indicies, axis=0)
    outputTraining  = np.delete(outputData, indicies, axis=0)
   
    return inputTraining, outputTraining, inputTest, outputTest, numberOfSymmFunc, outputs, parameters  
        
        
        
if __name__ == '__main__':
    readXYZ("../../LAMMPS_test/Silicon/Data/Si1000.xyz")
    
    
    
    
    
    