# read data from LAMMPS runs for training

import numpy as np
import sys
import symmetries
import readers
import symmetryParameters
import os
    


def SiTrainingData(dataFolder, symmFuncType, function=None, forces=False, Behler=True, 
                   klargerj=False, tags=True):
    """ 
    Coordinates and energies of neighbours is sampled from lammps
    Angular symmtry funcitons are used to transform input data  
    """
    
    filename = dataFolder + "neighbours.txt"
     
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
        parameters = symmetryParameters.SiBehler()
    else:
        parameters = symmetryParameters.SiBulkCustom()
        print
        print "Using customized parameters" 
                    
    numberOfSymmFunc = len(parameters)
    outputs = 1
    
    # decide on which symmetry parameters set to use
    sampleDir = filename[:-14]
    if Behler:
        if klargerj:
            print "k > j"
            symmetryFileName = sampleDir + 'symmetryBehlerklargerj.txt'
        else:
            print "k != j"
            symmetryFileName = sampleDir + 'symmetryBehlerkunequalj.txt'
            
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
    

def SiO2TrainingData(dataFolder, symmFuncType, atomType, forces=False):
    """ 
    Coordinates and energies of neighbours is sampled from lammps
    Angular symmtry funcitons are used to transform input data  
    """
    
    # read files
    if forces:
        print 'Forces included in lammps training data not implemented for SiO2'
        exit(1)
    else:
        print 'Forces are not included in lammps training data'
        if atomType == 0:
            print 'Training atom type 0: Si'
            x, y, z, r, types, E = readers.readNeighbourDataMultiType(dataFolder + 'neighbours0.txt')
        else:
            print 'Training atom type 1: O'
            x, y, z, r, types, E = readers.readNeighbourDataMultiType(dataFolder + 'neighbours1.txt')
    print "Lammps data is read..."
    
    if atomType == 0:
        parameters, elem2param = symmetryParameters.SiO2type0()
    else:
        parameters, elem2param = symmetryParameters.SiO2type1()
                    
    numberOfSymmFunc = len(parameters)
    outputs = 1
    
    symmetryFileName = 'symmetry%d.txt' % atomType
    symmetryFileName = dataFolder + symmetryFileName
    
    # apply symmetry or read already existing file
    if os.path.isfile(symmetryFileName):
        print "Reading symmetrized input data"
        inputData = readers.readSymmetryData(symmetryFileName)
        outputData = np.array(E)
        print "Energy is supplied from lammps"
    else: 
        # apply symmetry transformastion
        inputData, outputData = symmetries.applyThreeBodySymmetryMultiType(x, y, z, r, types, atomType,
                                                                           parameters, elem2param, symmFuncType, E=E, 
                                                                           sampleName=symmetryFileName)
        print 'Applying symmetry transformation'
    
    print inputData
    print outputData    
    
    # split in training set and test set randomly
    totalSize       = len(inputData)
    if len(inputData) < 10:
        testSize = 1
    else:
        testSize        = int(0.1*totalSize) 
    indicies        = np.random.choice(totalSize, testSize, replace=False)
    inputTest       = inputData[indicies]         
    outputTest      = outputData[indicies] 
    if totalSize > 1:
        inputTraining   = np.delete(inputData, indicies, axis=0)
        outputTraining  = np.delete(outputData, indicies, axis=0)
    else:
        inputTraining = inputTest
        outputTraining = outputTest
   
    return inputTraining, outputTraining, inputTest, outputTest, numberOfSymmFunc, outputs, parameters, elem2param
        
        
        
if __name__ == '__main__':
    pass
    
    
    
    
    
    