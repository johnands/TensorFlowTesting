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


def SiTrainingData(filename, symmFuncType):
    """ 
    Coordinates and energies of neighbours is sampled from lammps
    Angular symmtry funcitons are used to transform input data  
    """
    
    # read file
    x, y, z, r, E = readNeighbourData(filename)
    print "File is read..."
    
    # output data are energies for each neighbour list
    outputData = np.array(E)
    outputs = outputData.shape[1]
    
    # number of training vectors / neighbours lists
    size = len(x)

    # parameters G2
    widthsG2 = [0.04, 0.07, 0.11]
    centersG2 = [2.2, 2.8, 3.4]
    cutoffG2 = [6.0]

    # parameters G4
    thetaRangeG4 = [1, 4, 8]   # values..?
    cutoffG4 = [6.0]
    widthG4 = [0.04, 0.07, 0.1]
    inversionG4 = [-1.0, 1.0]
    
    #numberOfSymmFunc = len(widthsG2)*len(centersG2)*len(cutoffG2) *\
    #                   len(thetaRangeG4)*len(cutoffG4)*len(widthG4)*len(inversionG4) 
    numberOfSymmFunc = len(thetaRangeG4)*len(cutoffG4)*len(widthG4)*len(inversionG4) 
    inputData = np.zeros((size,numberOfSymmFunc))

    # loop through each data vector, i.e. each atomic environment
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
        ri = np.sqrt(ri)
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
            k = np.arange(len(ri[:])) != j
            rik = ri[k] 
            xik = xi[k]; yik = yi[k]; zik = zi[k]
            
            # compute angle and rjk
            argument = (xij*xik + yij*yik + zij*zik) / (rij*rik) 
            
            # floating-point error can yield an argument outside of arccos range
            if not (np.abs(argument) <= 1).all():
                for l, arg in enumerate(argument):
                    if arg < -1:
                        argument[l] = -1
                        print "Warning: %.14f has been replaced by %d" % (arg, argument[l])
                    if arg > 1:
                        argument[l] = 1
                        print "Warning: %.14f has been replaced by %d" % (arg, argument[l])
                        
            theta = np.arccos(argument)
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
            
            # G4
            for zeta in thetaRangeG4:
                for cutoff in cutoffG4:
                    for width in widthG4:
                        for inversion in inversionG4:
                            # find symmetry function value for triplets (i,j,k) for all k
                            inputData[i,symmFuncNumber] += symmetryFunctions.G4(rij, rik, rjk, theta, \
                                                                                zeta, width, cutoff, inversion)
                            symmFuncNumber += 1
            
            # G2
            """for width in widthsG2:
                for center in centersG2:
                    for cutoff in cutoffG2:
                        inputData[i,symmFuncNumber] += symmetryFunctions.G2(rij, cutoff, width, center)
                        symmFuncNumber += 1

                            
            # G2 x G4
            for width2 in widthsG2:
                for center2 in centersG2:
                    for cutoff2 in cutoffG2:
                        for zeta4 in thetaRangeG4:
                            for cutoff4 in cutoffG4:
                                for width4 in widthG4:
                                    for inversion4 in inversionG4:
                                        # find symmetry function value for triplets (i,j,k) for all k
                                        inputData[i,symmFuncNumber] += symmetryFunctions.G4(rij, rik, rjk, theta, \
                                                                                            zeta4, width4, cutoff4, inversion4) * \
                                                                       symmetryFunctions.G2(rij, cutoff2, width2, center2) 
                                        symmFuncNumber += 1"""
 

        
        # count zeros
        fractionOfNonZeros += np.count_nonzero(inputData[i,:]) / float(numberOfSymmFunc)
        if not np.any(inputData[i,:]):
            fractionOfInputVectorsOnlyZeros += 1
            print inputData[i,:]
            
        # show progress
        sys.stdout.write("\r%2d %% complete" % ((float(i)/size)*100))
        sys.stdout.flush()
      
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
    
    print "Input data:"
    maxInput = np.max(inputData)
    minInput = np.min(inputData)
    print "Max: ", maxInput
    print "Min: ", minInput
    print "Mean: ", np.mean(inputData)
    
    # shift inputs so that average is zero
    inputData = inputData - np.mean(inputData)
    
    # scale the covariance
    C = np.sum(inputData**2, axis=0) / float(size)
    print C
    inputData = inputData + (1 - np.sqrt(C))/float(numberOfSymmFunc)
    C = np.sum(inputData**2, axis=0) / float(size)
    print C
    #inputData = inputData + (1 - )
    exit(1)
    
    # normalize to [-1,1]
    #inputData = 2 * (inputData - minInput) / (maxInput - minInput) - 1 
    
        
    print "Normalized input data:"
    maxInput = np.max(inputData)
    minInput = np.min(inputData)
    print "Max: ", maxInput
    print "Min: ", minInput
    print "Mean: ", np.mean(inputData)
    
    print "Output data:"
    maxOutput = np.max(outputData)
    minOutput = np.min(outputData)
    print "Max: ", maxOutput
    print "Min: ", minOutput
    print "Mean: ", np.mean(outputData)
    
    # normalize output data
    outputData = 2 * (outputData - minOutput) / (maxOutput - minOutput) - 1
    
    print "Normalized output data:"
    maxOutput = np.max(outputData)
    minOutput = np.min(outputData)
    print "Max: ", maxOutput
    print "Min: ", minOutput
    print "Mean: ", np.mean(outputData)
    
    return inputTraining, outputTraining, inputTest, outputTest, numberOfSymmFunc, outputs      




        
if __name__ == '__main__':
    readXYZ("../../LAMMPS_test/Silicon/Data/Si1000.xyz")