# read data from LAMMPS runs for training

import numpy as np
import sys
import symmetryFunctions

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
    
    
def SiTrainingData(function, filename, neighbours, symmFuncType, \
                   low, high, outputs=1):
        
    cutoff = high
    x, y, z, r = readXYZ(filename, cutoff)
    size = len(x)

    outputData = np.zeros((size, outputs))
    
    # generate symmetry function input data
    thetaRange = [1, 2, 4]   # values..?
    cutoff = [6.0]
    widths = [0.01, 0.025, 0.05, 0.07, 0.1]
    inversions = [-1.0, 1.0]
    
    numberOfSymmFunc = len(thetaRange)*len(cutoff)*len(widths)*len(inversions)    
    inputData = np.zeros((size,numberOfSymmFunc))

    # loop through each r vector, i.e. each atomic environment
    thetaMax = 0.0
    thetaMin = 100.0
    fractionOfNonZeros = 0.0
    fractionOfInputVectorsOnlyZeros = 0.0
    meanNeighbours = 0.0;
    for i in xrange(size):      
    
        # neighbour coordinates for atom i
        xi = np.array(x[i][:])
        yi = np.array(y[i][:])
        zi = np.array(z[i][:])
        ri = np.array(r[i][:])
        numberOfNeighbours = len(xi)
        
        # count mean number of neighbours
        meanNeighbours += numberOfNeighbours
        
        # sum over all neighbours k for each neighbour j
        # this loop takes care of both 2-body and 3-body configs   
        for j in xrange(numberOfNeighbours-1):
                      
            # atom j
            rij = ri[j]
            xij = xi[j]; yij = yi[j]; zij = zi[j]
            
            # all k != i,j OR I > J ???
            k = np.arange(len(ri[:])) > j
            rik = ri[k] 
            xik = xi[k]; yik = yi[k]; zik = zi[k]
            
            # compute angle and rjk
            theta = np.arccos( (xij*xik + yij*yik + zij*zik) / (rij*rik) )
            rjk = np.sqrt( rij**2 + rik**2 - 2*rij*rik*np.cos(theta) )
            
            # check max and min
            Max = np.max(theta)
            Min = np.min(theta)
            if Max > thetaMax:
                thetaMax = Max
            if Min < thetaMin:
                thetaMin = Min
            
            # find value of each symmetry function for this triplet
            symmFuncNumber = 0
            for angle in thetaRange:
                for width in widths:
                    for inversion in inversions:
                        # find symmetry function value for triplets (i,j,k) for all k
                        inputData[i,symmFuncNumber] += symmetryFunctions.G4(rij, rik, rjk, theta, \
                                                                            angle, width, cutoff, inversion)
                        symmFuncNumber += 1
                                           
            # 3-body, rik and theta are vectors
            outputData[i,0] += np.sum(function(rij, rik, theta))
        
        # count zeros
        fractionOfNonZeros += np.count_nonzero(inputData[i,:]) / float(numberOfSymmFunc)
        if not np.any(inputData[i,:]):
            fractionOfInputVectorsOnlyZeros += 1
            print inputData[i,:]
      
    # split in training set and test set
    split = int(0.8*size)    
    inputTraining  = inputData[:split,:]
    outputTraining = outputData[:split,:]
    inputTest      = inputData[split:,:]         
    outputTest     = outputData[split:,:]
    
    print   
    print "max theta: ", thetaMax
    print "min theta: ", thetaMin
    print
    
    fractionOfZeros = 1 - fractionOfNonZeros / float(size)
    fractionOfInputVectorsOnlyZeros /= float(size)
    print "Fraction of zeros: ", fractionOfZeros
    print "Fraction of input vectors with only zeros: ", fractionOfInputVectorsOnlyZeros
    print
    
    meanNeighbours /= float(size)
    print "Mean number of neighbours: ", meanNeighbours
    print
    
    print "Output data:"
    maxInput = np.max(inputData)
    minInput = np.min(inputData)
    print "Max: ", maxInput
    print "Min: ", minInput
    print "Mean: ", np.mean(inputData)
    
    # normalize to [-1,1]
    #inputData = 2 * (inputData - minValue) / (maxValue - minValue) - 1 
    
    print "Output data:"
    maxOutput = np.max(outputData)
    minOutput = np.min(outputData)
    print "Max: ", maxOutput
    print "Min: ", minOutput
    print "Mean: ", np.mean(outputData)
    
    # normalize output data
    #outputData = 2 * (outputData - minOutput) / (maxOutput - minOutput) - 1 
    
    return inputTraining, outputTraining, inputTest, outputTest




        
if __name__ == '__main__':
    readXYZ("../../LAMMPS_test/Silicon/Data/Si1000.xyz")