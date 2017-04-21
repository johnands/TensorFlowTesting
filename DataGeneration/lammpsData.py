# read data from LAMMPS runs for training

import numpy as np
import sys
import symmetries
import os

def readXYZ(filename, cut):
    """
    Make four nested lists x, y, z, r where vector x[i][:]
    is the x-coordinates of all the neighbours of atom i
    """
    
    # process xyz-file
    with open(filename, 'r') as inFile:
    
        # skip three first lines
        for _ in xrange(3):
            inFile.readline()
            
        numberOfAtoms = int(inFile.readline())
        print "Number of atoms: ", numberOfAtoms
        
        inFile.readline()
        
        systemSize = []
        for _ in xrange(3):
            systemSize.append( float(inFile.readline().split()[1]) )
        systemSize = np.array(systemSize)
        systemSizeHalf = systemSize / 2.0
        print "System size: ", systemSize
        print "System size half: ", systemSizeHalf
        
        inFile.readline()
        
        # read positions and store in array
        positions = np.zeros((numberOfAtoms, 3))   
        counter = 0     
        for line in inFile:
            coordinates = line.split()
            positions[counter,0] = float(coordinates[0])
            positions[counter,1] = float(coordinates[1])
            positions[counter,2] = float(coordinates[2])
            counter += 1
        
    # make neighbour list for all atoms in the system
    x = []; y = []; z = []; r = []
    for i in xrange(numberOfAtoms):
        atom1 = positions[i]
        xNeigh = []; yNeigh = []; zNeigh = []; rNeigh = []
        for j in xrange(numberOfAtoms):
            if i != j:
                atom2 = positions[j]
                dr = atom1 - atom2 
                
                # periodic boundary conditions
                for dim in xrange(3):
                    if dr[dim] > systemSizeHalf[dim]:
                        dr[dim] -= systemSize[dim]
                    elif dr[dim] < -systemSizeHalf[dim]:
                        dr[dim] += systemSize[dim]
                    
                distance = np.sqrt( dr[0]**2 + dr[1]**2 + dr[2]**2 )
                
                # add to neighbour lists if r2 under cut
                if distance < cut:
                    xNeigh.append(dr[0])
                    yNeigh.append(dr[1])
                    zNeigh.append(dr[2])
                    rNeigh.append(distance)
                    
        x.append(xNeigh)
        y.append(yNeigh)
        z.append(zNeigh)
        r.append(rNeigh)
  
        # show progress
        sys.stdout.write("\r%2d %% complete" % ((float(i)/numberOfAtoms)*100))
        sys.stdout.flush()

    return x, y, z, r
    
    
def readNeighbourData(filename):
    
    with open(filename, 'r') as inFile:
        
        x = []; y = []; z = []
        r = []; E = []
        for line in inFile:
            words = line.split()
            N = (len(words) - 1) / 4
            xi = []; yi = []; zi = [];
            ri = [];
            for i in xrange(N):
                xi.append(float(words[4*i]))
                yi.append(float(words[4*i+1]))
                zi.append(float(words[4*i+2]))
                ri.append(float(words[4*i+3]))
                
            x.append(xi)
            y.append(yi)
            z.append(zi)
            r.append(ri)
            E.append([float(words[-1])])  
                      
    return x, y, z, r, E
    
def readEnergy(filename):
    
    with open(filename, 'r') as inFile:
        
        E = []
        for line in inFile:
            words = line.split()
            E.append([float(words[-1])])  
                      
    return E
    

def readNeighbourDataForce(filename):
    
    with open(filename, 'r') as inFile:
        
        x = []; y = []; z = []; r = [];
        E = []; Fx = []; Fy = []; Fz = []
        for line in inFile:
            words = line.split()

            N = (len(words) - 4) / 4
            xi = []; yi = []; zi = [];
            ri = [];
            for i in xrange(N):
                xi.append(float(words[4*i]))
                yi.append(float(words[4*i+1]))
                zi.append(float(words[4*i+2]))
                ri.append(float(words[4*i+3]))
                
            x.append(xi)
            y.append(yi)
            z.append(zi)
            r.append(ri)
            E.append([float(words[-4])])  
            Fx.append([float(words[-3])])
            Fy.append([float(words[-2])])
            Fz.append([float(words[-1])])
            
    return x, y, z, r, E, Fx, Fy, Fz
    

def readNeighbourDataForceTag(filename):
    
    with open(filename, 'r') as inFile:
        
        x = []; y = []; z = []; r = [];
        tags = []
        E = []; Fx = []; Fy = []; Fz = []
        for line in inFile:
            words = line.split()

            N = (len(words) - 4) / 5
            xi = []; yi = []; zi = [];
            tagsi = []
            ri = [];
            for i in xrange(N):
                tagsi.append(float(words[5*i]))
                xi.append(float(words[5*i+1]))
                yi.append(float(words[5*i+2]))
                zi.append(float(words[5*i+3]))
                ri.append(float(words[5*i+4]))
                
            tags.append(tagsi)
            x.append(xi)
            y.append(yi)
            z.append(zi)
            r.append(ri)
            E.append([float(words[-4])])  
            Fx.append([float(words[-3])])
            Fy.append([float(words[-2])])
            Fz.append([float(words[-1])])
            
    return x, y, z, r, E, Fx, Fy, Fz, tags
    
    
    
def readEnergyAndForce(filename):
    
    with open(filename, 'r') as inFile:
        
        E = []; Fx = []; Fy = []; Fz = []
        for line in inFile:
            words = line.split()
            E.append([float(words[-4])])  
            Fx.append([float(words[-3])])
            Fy.append([float(words[-2])])
            Fz.append([float(words[-1])])            
            
    return E, Fx, Fy, Fz
    
    
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
    cutoff = 6.0
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
            
    
    
def readSymmetryData(filename):
    
    inputData = []
    with open(filename, 'r') as infile:
        
        for line in infile:
            inputVector = []
            words = line.split()
            for word in words:
                inputVector.append(float(word))
            inputData.append(inputVector)
            
    return np.array(inputData)


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
            x, y, z, r, E, Fx, Fy, Fz, _ = readNeighbourDataForceTag(filename)
        else:
            print 
            print "Tags are not included in neighbour lists"
            x, y, z, r, E, Fx, Fy, Fz = readNeighbourDataForce(filename)
        Fx = np.array(Fx)
        Fy = np.array(Fy)
        Fz = np.array(Fz)
        print "Forces is applied"
    else:
        x, y, z, r, E = readNeighbourData(filename)
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
            symmetryFileName = sampleDir + 'symmetryCustomklargerj.txt'
        else:
            print "k != j"
            symmetryFileName = sampleDir + 'symmetryCustomkunequalj.txt'
        print "Using customized symmetry parameters"
            
    # apply symmetry or read already existing file
    if os.path.isfile(symmetryFileName):
        print "Reading symmetrized input data"
        inputData = readSymmetryData(symmetryFileName)
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
    x, y, z, r, E = readNeighbourData(filename)
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
    
    
    
    
    
    