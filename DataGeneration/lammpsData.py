# read data from LAMMPS runs for training

import numpy as np
import sys
import symmetries

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


def SiTrainingData(filename, symmFuncType, function=None):
    """ 
    Coordinates and energies of neighbours is sampled from lammps
    Angular symmtry funcitons are used to transform input data  
    """
    
    # read file
    x, y, z, r, E = readNeighbourData(filename)
    print "File is read..."

    # parameters G2
    widthG2 = [0.001, 0.01, 0.1, 1]
    cutoffG2 = [4.0]
    centerG2 = [0.0]

    # parameters G4
    widthG4 = [0.001, 0.01]      
    cutoffG4 = [4.0]
    thetaRangeG4 = [1, 2, 4] 
    inversionG4 = [1.0, -1.0]
    
    # make nested list of all symetry function parameters
    parameters = []
    for width in widthG2:
        for cutoff in cutoffG2:
            for center in centerG2:           
                parameters.append([width, cutoff, center])
             
    for width in widthG4:   
        for cutoff in cutoffG4:
            for zeta in thetaRangeG4:
                for inversion in inversionG4:
                    parameters.append([width, cutoff, zeta, inversion])
                    
    numberOfSymmFunc = len(parameters)
    outputs = 1
                   
    # apply symmetry transformastion
    inputData, outputData = symmetries.applyThreeBodySymmetry(x, y, z, r, parameters, function=function, E=E)
    
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
    
    
    
    
    
    