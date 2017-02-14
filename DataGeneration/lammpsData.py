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


def SiTrainingData(filename, symmFuncType, function=None):
    """ 
    Coordinates and energies of neighbours is sampled from lammps
    Angular symmtry funcitons are used to transform input data  
    """
    
    # read file
    x, y, z, r, E = readNeighbourData(filename)
    print "File is read..."
    
    # number of training vectors / neighbours lists
    size = len(x)
    
    # output data are energies for each neighbour list or SW
    if function == None:    
        outputData = np.array(E)
    else:
        outputData = np.zeros((size, 1))
        
    outputs = outputData.shape[1]

    # parameters G2
    widthG2 = [0.001, 0.01, 0.1, 1]
    cutoffG2 = [4.0]
    centerG2 = [0.0]

    # parameters G4
    widthG4 = [0.001, 0.01]      
    cutoffG4 = [4.0]
    thetaRangeG4 = [1, 2, 4] 
    inversionG4 = [1.0, -1.0]
    
    numberOfSymmFunc = len(widthG2)*len(centerG2)*len(cutoffG2) + \
                       len(thetaRangeG4)*len(cutoffG4)*len(widthG4)*len(inversionG4) 
    #numberOfSymmFunc = len(thetaRangeG4)*len(cutoffG4)*len(widthG4)*len(inversionG4) 
    
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

    inputData = np.zeros((size,numberOfSymmFunc))

    # loop through each data vector, i.e. each atomic environment
    #thetaMax = 0.0
    #thetaMin = 100.0
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
        for j in xrange(numberOfNeighbours):
                      
            # atom j
            rij = ri[j]
            xij = xi[j]; yij = yi[j]; zij = zi[j]
            
            # all k != i,j OR I > J ???
            k = np.arange(len(ri[:])) > j  
            rik = ri[k] 
            xik = xi[k]; yik = yi[k]; zik = zi[k]
            
            # compute angle and rjk
            cosTheta = (xij*xik + yij*yik + zij*zik) / (rij*rik) 
            
            # floating-point error can yield an argument outside of arccos range
            if not (np.abs(cosTheta) <= 1).all():
                for l, arg in enumerate(cosTheta):
                    if arg < -1:
                        cosTheta[l] = -1
                        print "Warning: %.14f has been replaced by %d" % (arg, cosTheta[l])
                    if arg > 1:
                        cosTheta[l] = 1
                        print "Warning: %.14f has been replaced by %d" % (arg, cosTheta[l])
                        
            #theta = np.arccos(argument)
            #rjk = np.sqrt( rij**2 + rik**2 - 2*rij*rik*np.cos(theta) )
            
            rjk = np.sqrt( rij**2 + rik**2 - 2*rij*rik*cosTheta )
            
            # check max and min
            """Max = np.max(theta)
            Min = np.min(theta)
            if Max > thetaMax:
                thetaMax = Max
            if Min < thetaMin:
                thetaMin = Min"""
            
            # find value of each symmetry function for this triplet
            symmFuncNumber = 0

            # G2
            for width in widthG2:
                for cutoff in cutoffG2:
                    for center in centerG2:
                        inputData[i,symmFuncNumber] += symmetryFunctions.G2(rij, width, cutoff, center)
                        symmFuncNumber += 1
                        
            # G4
            for width in widthG4:
                for cutoff in cutoffG4:             
                    for zeta in thetaRangeG4:
                        for inversion in inversionG4:
                            # find symmetry function value for triplets (i,j,k) for all k
                            inputData[i,symmFuncNumber] += symmetryFunctions.G4(rij, rik, rjk, cosTheta, \
                                                                                width, cutoff, zeta, inversion)
                            symmFuncNumber += 1
        
            # calculate energy with my S-W-potential, not use lammps energy
            if function != None:
                outputData[i,0] += np.sum( function(rij, rik, cosTheta) )
            
        # shuffle input vector
        #np.random.shuffle(inputData[i,:])
        
        # count zeros
        fractionOfNonZeros += np.count_nonzero(inputData[i,:]) / float(numberOfSymmFunc)
        if not np.any(inputData[i,:]):
            fractionOfInputVectorsOnlyZeros += 1
            print inputData[i,:]
            
        # show progress
        sys.stdout.write("\r%2d %% complete" % ((float(i)/size)*100))
        sys.stdout.flush()
    
    """print   
    print "max theta: ", thetaMax
    print "min theta: ", thetaMin
    print"""
    
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
    print "Mean: ", np.mean(inputData,)
    print 
    
    print "Output data:"
    maxOutput = np.max(outputData)
    minOutput = np.min(outputData)
    print "Max: ", maxOutput
    print "Min: ", minOutput
    print "Mean: ", np.mean(outputData)
    print
    
    # normalize to [-1,1]
    if normalizeFlag:
        inputData = 2 * (inputData - minInput) / (maxInput - minInput) - 1 
        outputData = 2 * (outputData - minOutput) / (maxOutput - minOutput) - 1    
    
    if shiftMeanFlag:
        # shift inputs so that average is zero
        inputData  -= np.mean(inputData, axis=0)
        # shift outputs so that average is zero
        outputData -= np.mean(outputData, axis=0)
    
    # scale the covariance
    if scaleCovarianceFlag:
        C = np.sum(inputData**2, axis=0) / float(size)
        print C
    
    if decorrelateFlag:     
        # decorrelate data
        cov = np.dot(inputData.T, inputData) / inputData.shape[0] # get the data covariance matrix
        U,S,V = np.linalg.svd(cov)
        inputData = np.dot(inputData, U) # decorrelate the data
        C = np.sum(inputData**2, axis=0) / float(size)
        print C
          
    print "New input data:"
    maxInput = np.max(inputData)
    minInput = np.min(inputData)
    print "Max: ", maxInput
    print "Min: ", minInput
    print "Mean: ", np.mean(inputData)
    print
    
    print "New output data:"
    maxOutput = np.max(outputData)
    minOutput = np.min(outputData)
    print "Max: ", maxOutput
    print "Min: ", minOutput
    print "Mean: ", np.mean(outputData)
    print  
    
    # split in training set and test set
    testSize        = int(0.1*size) 
    indicies        = np.random.choice(size, testSize, replace=False)
    inputTest       = inputData[indicies]         
    outputTest      = outputData[indicies] 
    inputTraining   = np.delete(inputData, indicies, axis=0)
    outputTraining  = np.delete(outputData, indicies, axis=0)
   
    return inputTraining, outputTraining, inputTest, outputTest, numberOfSymmFunc, outputs, parameters  



normalizeFlag       = False    
shiftMeanFlag       = False
scaleCovarianceFlag = False
decorrelateFlag     = False
        
        
        
if __name__ == '__main__':
    readXYZ("../../LAMMPS_test/Silicon/Data/Si1000.xyz")
    
    
    
    
    
    