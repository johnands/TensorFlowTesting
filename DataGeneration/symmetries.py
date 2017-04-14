import numpy as np
import tensorflow as tf
import sys
import symmetryFunctions
     
           
           
def applyTwoBodySymmetry(inputTemp, parameters):
    """
    Transform input coordinates with 2-body symmetry functions
    Input coordinates can be random or sampled from lammps
    Output data is generated beforehand
    """
    
    size = len(inputTemp)
    numberOfSymmFunc = len(parameters)
    
    inputData = np.zeros((size,numberOfSymmFunc))

    # transform input data
    fractionOfNonZeros = 0.0
    fractionOfInputVectorsOnlyZeros = 0.0
    for i in xrange(size):
        
        # atomic environment of atom i
        rij = np.array(inputTemp[i][:]) 
        
        # find value of each symmetry function for this triplet
        symmFuncNumber = 0
        for s in parameters:
            if len(s) == 1:
                inputData[i,symmFuncNumber] += symmetryFunctions.G1(rij, s[0])              
            else:
                inputData[i,symmFuncNumber] += symmetryFunctions.G2(rij, s[0], s[1], s[2])
            symmFuncNumber += 1
               
        # count zeros
        fractionOfNonZeros += np.count_nonzero(inputData[i,:]) / float(numberOfSymmFunc)
        if not np.any(inputData[i,:]):
            fractionOfInputVectorsOnlyZeros += 1
            
        # show progress
        sys.stdout.write("\r%2d %% complete" % ((float(i)/size)*100))
        sys.stdout.flush()
        
    fractionOfZeros = 1 - fractionOfNonZeros / float(size)
    fractionOfInputVectorsOnlyZeros /= float(size)
    print "Fraction of zeros: ", fractionOfZeros
    print "Fraction of input vectors with only zeros: ", fractionOfInputVectorsOnlyZeros
    
    return inputData
    
    
def calculateForces(x, y, z, r, parameters, forceFile, dEdG):
    
    print
    print "Computing forces..."
    
    size = len(x)
    
    with open(forceFile, 'w') as outfile:
        
        # calculate force for each neighbour atom at each time step in given configs
        for i in xrange(size):  
        
            # neighbour coordinates for atom i
            xi = np.array(x[i][:])
            yi = np.array(y[i][:])
            zi = np.array(z[i][:])
            ri = np.array(r[i][:])
            ri = np.sqrt(ri)
            numberOfNeighbours = len(xi)
               
            # sum over all neighbours k for each neighbour j
            # this loop takes care of both 2-body and 3-body configs
            # want to save total force on all NEIGHBOURS, not the atom i itself
            Fx = np.zeros(numberOfNeighbours)
            Fy = np.zeros(numberOfNeighbours)
            Fz = np.zeros(numberOfNeighbours)
            for j in xrange(numberOfNeighbours):
                
                # atom j
                rij = ri[j]
                xij = xi[j]; yij = yi[j]; zij = zi[j]
                
                # all k != i,j OR I > J ??? REMEMBER TO CHANGE WHEN NEEDED
                k = np.arange(len(ri[:])) > j
                rik = ri[k] 
                xik = xi[k]; yik = yi[k]; zik = zi[k]
                
                # compute cos(theta_ijk) and rjk
                cosTheta = (xij*xik + yij*yik + zij*zik) / (rij*rik) 
                
                xjk = xij - xik
                yjk = yij - yik
                zjk = zij - zik
                rjk = np.sqrt(xjk*xjk + yjk*yjk + zjk*zjk) 
                
                # differentiate each symmetry function and compute forces for current input vector
                # xij, yij, zij, rij are numbers, i.e. on neighbour j at a tie
                # xik, yik, zik, rik are vectors, i.e. all neighbours k > j
                # we therefore include all triplets for a specific pair (i,j)
                symmFuncNumber = 0
                for s in parameters:
                    if len(s) == 3:
                        # compute derivative of G2 w.r.t. (x,y,z) of all neighbours simultaneously
                        dij = symmetryFunctions.dG2dr(xij, yij, zij, rij, s[0], s[1], s[2]) 
                        Fx[j] += dEdG[i][symmFuncNumber]*dij[0]
                        Fy[j] += dEdG[i][symmFuncNumber]*dij[1]
                        Fz[j] += dEdG[i][symmFuncNumber]*dij[2]
                        
                    else:
                        # compute derivative of G5 w.r.t. (x,y,z) of all k for on j 
                        # atom j must get contribution from all k's
                        # each k gets one contribution per k
                        dij3 = symmetryFunctions.dG5dj(xij, yij, zij, xik, yik, zik, rij, rik, cosTheta, \
                                                      s[0], s[1], s[2], s[3])
                        dik = symmetryFunctions.dG5dk(xij, yij, zij, xik, yik, zik, rij, rik, cosTheta, \
                                                      s[0], s[1], s[2], s[3])
                                                      
                        # atom j gets force contribution for all ks, i.e. all triplets for this (i,j)                                      
                        Fx[j] += dEdG[i][symmFuncNumber]*np.sum(dij3[0])
                        Fy[j] += dEdG[i][symmFuncNumber]*np.sum(dij3[1])
                        Fz[j] += dEdG[i][symmFuncNumber]*np.sum(dij3[2])
                        
                        # find force contribution on each k
                        Fx[k] += dEdG[i][symmFuncNumber]*dik[0]
                        Fy[k] += dEdG[i][symmFuncNumber]*dik[1]
                        Fz[k] += dEdG[i][symmFuncNumber]*dik[2]
                                         
                    symmFuncNumber += 1
                
                # write force on current j, this will be total force when k > j
                outfile.write('%.17g %.17g %.17g ' % (Fx[j], Fy[j], Fz[j])) 
            
                    
            outfile.write('\n')
            
            # show progress
            sys.stdout.write("\r%2d %% complete" % ((float(i)/size)*100))
            sys.stdout.flush()
            
            
def calculateForcesTags(x, y, z, r, tags, parameters, forceFile, dEdG, chosenAtom, numberOfAtoms):
    
    print
    print "Computing forces..."
    
    size = len(x)    
    atom = 0
    
    with open(forceFile, 'w') as outfile:
        
        # calculate force for each neighbour atom at each time step in given configs
        for i in xrange(size):
            
            # reset atom counter, this means new time step
            if atom == numberOfAtoms-1:
                atom = 0
                outfile.write('\n')
            
            # skip if current neighbour list belongs to chosen atom
            # can I just use all tags as strings?
            if tags[i][-1] == chosenAtom:
                atom += 1
                continue
            
            # find out which coordinates in current list that belongs to chosen atom
            chosen = np.where(tags[i] == chosenAtom)[0]
            
            # skip if chosen atom is not in this list
            if len(chosen) == 0:
                atom += 1
                continue
        
            # neighbour coordinates for atom i
            xi = np.array(x[i][:])
            yi = np.array(y[i][:])
            zi = np.array(z[i][:])
            ri = np.array(r[i][:])
            ri = np.sqrt(ri)
            numberOfNeighbours = len(xi)
                         
            # sum over all neighbours k for each neighbour j
            # this loop takes care of both 2-body and 3-body configs
            # want to save total force on all NEIGHBOURS, not the atom i itself
            Fx = np.zeros(numberOfNeighbours)
            Fy = np.zeros(numberOfNeighbours)
            Fz = np.zeros(numberOfNeighbours)
            for j in xrange(numberOfNeighbours):
                
                # atom j
                rij = ri[j]
                xij = xi[j]; yij = yi[j]; zij = zi[j]
                
                # all k != i,j OR I > J ??? REMEMBER TO CHANGE WHEN NEEDED
                k = np.arange(len(ri[:])) > j
                rik = ri[k] 
                xik = xi[k]; yik = yi[k]; zik = zi[k]
                
                # skip if chosen atom is not in current triplet
                if not (j == chosen or (k == chosen).any()):
                    continue
                    
                
                # compute cos(theta_ijk) and rjk
                cosTheta = (xij*xik + yij*yik + zij*zik) / (rij*rik) 
                
                xjk = xij - xik
                yjk = yij - yik
                zjk = zij - zik
                rjk = np.sqrt(xjk*xjk + yjk*yjk + zjk*zjk) 
                
                # differentiate each symmetry function and compute forces for current input vector
                # xij, yij, zij, rij are numbers, i.e. on neighbour j at a tie
                # xik, yik, zik, rik are vectors, i.e. all neighbours k > j
                # we therefore include all triplets for a specific pair (i,j)
                symmFuncNumber = 0
                for s in parameters:
                    if len(s) == 3:
                        # compute derivative of G2 w.r.t. (x,y,z) of all neighbours simultaneously
                        dij = symmetryFunctions.dG2dr(xij, yij, zij, rij, s[0], s[1], s[2]) 
                        Fx[j] += dEdG[i][symmFuncNumber]*dij[0]
                        Fy[j] += dEdG[i][symmFuncNumber]*dij[1]
                        Fz[j] += dEdG[i][symmFuncNumber]*dij[2]
                        
                    else:
                        # compute derivative of G5 w.r.t. (x,y,z) of all k for on j 
                        # atom j must get contribution from all k's
                        # each k gets one contribution per k
                        dij3 = symmetryFunctions.dG5dj(xij, yij, zij, xik, yik, zik, rij, rik, cosTheta, \
                                                      s[0], s[1], s[2], s[3])
                        dik = symmetryFunctions.dG5dk(xij, yij, zij, xik, yik, zik, rij, rik, cosTheta, \
                                                      s[0], s[1], s[2], s[3])
                                                      
                        # atom j gets force contribution for all ks, i.e. all triplets for this (i,j)                                      
                        Fx[j] += dEdG[i][symmFuncNumber]*np.sum(dij3[0])
                        Fy[j] += dEdG[i][symmFuncNumber]*np.sum(dij3[1])
                        Fz[j] += dEdG[i][symmFuncNumber]*np.sum(dij3[2])
                        
                        # find force contribution on each k
                        Fx[k] += dEdG[i][symmFuncNumber]*dik[0]
                        Fy[k] += dEdG[i][symmFuncNumber]*dik[1]
                        Fz[k] += dEdG[i][symmFuncNumber]*dik[2]
                                         
                    symmFuncNumber += 1
                
                # write force on current j, this will be total force when k > j
                outfile.write('%.17g %.17g %.17g ' % (Fx[j], Fy[j], Fz[j])) 
            
                    
            outfile.write('\n')
            
            # show progress
            sys.stdout.write("\r%2d %% complete" % ((float(i)/size)*100))
            sys.stdout.flush()
      
      
           
def applyThreeBodySymmetry(x, y, z, r, parameters, symmFuncType, function=None, E=None, forces=False,
                           sampleName='', klargerj=False):
    """
    Transform input coordinates with 2- and 3-body symmetry functions
    Input coordinates can be random or sampled from lammps
    Output data can be supplied with an array E or be generated
    using the optional function argument
    """
    
    if symmFuncType == 'G4':
        print 
        print 'Using G4'
    elif symmFuncType == 'G5':
        print 
        print 'Using G5'
    else:
        print 'Not valid triplet symmetry function'
        exit(1)
        
    if klargerj:
        print 
        print "Using k > j when training"
    else:
        print
        print "Using k != j when training"
    
    size = len(x)
    numberOfSymmFunc = len(parameters)
    
    inputData  = np.zeros((size,numberOfSymmFunc)) 
       
    if function == None:
        if E == None:
            print "Either function or energy must be supplied"
            exit(1)
        else:
            outputData = np.array(E)
            print "Energy is supplied from lammps"
    else:
        outputData = np.zeros((size, 1))
        print "Energy is generated with user-supplied function"
        
    # store 
    Rijs = []
    drijs = []
    Riks = []
    driks = []
    cosThetas = []
    Rjks = []
    drjks = []
    
    # loop through each data vector, i.e. each atomic environment
    fractionOfNonZeros = 0.0
    fractionOfInputVectorsOnlyZeros = 0.0
    meanNeighbours = 0.0
    rjkMin = 100.0
    rjkMax = 0.0
    for i in xrange(size):  
        
        # nested lists
        drijs.append([])
        Riks.append([])
        driks.append([])
        cosThetas.append([])
        Rjks.append([])
        drjks.append([])
    
        # neighbour coordinates for atom i
        xi = np.array(x[i][:])
        yi = np.array(y[i][:])
        zi = np.array(z[i][:])
        ri = np.array(r[i][:])
        ri = np.sqrt(ri)
        numberOfNeighbours = len(xi)
        
        # store
        Rijs.append(ri)     # list of arrays
        drijs[i].append(xi) # list of list of arrays
        drijs[i].append(yi)
        drijs[i].append(zi)
        
        # count mean number of neighbours
        meanNeighbours += numberOfNeighbours
        
        # sum over all neighbours k for each neighbour j
        # this loop takes care of both 2-body and 3-body configs   
        for j in xrange(numberOfNeighbours):
            
            # 3d lists
            driks[i].append([]);
            drjks[i].append([]);
                      
            # atom j
            rij = ri[j]
            xij = xi[j]; yij = yi[j]; zij = zi[j]
            
            # all k != i,j OR I > J
            if klargerj:
                k = np.arange(len(ri[:])) > j
            else:
                k = np.arange(len(ri[:])) != j  
                
            rik = ri[k] 
            xik = xi[k]; yik = yi[k]; zik = zi[k]
            
            # compute cos(theta_ijk) and rjk
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
            
            xjk = xij - xik
            yjk = yij - yik
            zjk = zij - zik
            rjk = np.sqrt(xjk*xjk + yjk*yjk + zjk*zjk)
            
            # store
            Riks[i].append(rik)             # list of list of arrays
            driks[i][j].append(xik)         # list of list of list of arrays
            driks[i][j].append(yik)
            driks[i][j].append(zik)
            cosThetas[i].append(cosTheta)
            Rjks[i].append(rjk)
            drjks[i][j].append(xjk)
            drjks[i][j].append(yjk)
            drjks[i][j].append(zjk)
                      
            if rjk.size > 0:            
                minR = np.min(rjk)
                maxR = np.max(rjk)
                if minR < rjkMin:
                    rjkMin = minR
                if maxR > rjkMax:
                    rjkMax = maxR
            
            # find value of each symmetry function for this triplet
            symmFuncNumber = 0
            for s in parameters:
                if len(s) == 3:
                    inputData[i,symmFuncNumber] += symmetryFunctions.G2(rij, s[0], s[1], s[2])
                else:
                    if symmFuncType == 'G4':
                        inputData[i,symmFuncNumber] += symmetryFunctions.G4(rij, rik, rjk, cosTheta, \
                                                          s[0], s[1], s[2], s[3])
                    else:
                        inputData[i,symmFuncNumber] += symmetryFunctions.G5(rij, rik, cosTheta, \
                                                          s[0], s[1], s[2], s[3])
                    
                symmFuncNumber += 1

            # calculate energy with supplied 3-body function or with
            if function != None:
                outputData[i,0] += function(rij, rik, cosTheta)                   
        
        # count zeros
        fractionOfNonZeros += np.count_nonzero(inputData[i,:]) / float(numberOfSymmFunc)
        if not np.any(inputData[i,:]):
            fractionOfInputVectorsOnlyZeros += 1
            print i
            
        # show progress
        sys.stdout.write("\r%2d %% complete" % ((float(i)/size)*100))
        sys.stdout.flush()
        
        
    if sampleName:
        print 
        print "Writing symmetrized input data to file"
        with open(sampleName, 'w') as outfile:
            for vector in inputData:
                for symmValue in vector:
                    outfile.write('%g ' % symmValue)
                outfile.write('\n')
                  
    fractionOfZeros = 1 - fractionOfNonZeros / float(size)
    fractionOfInputVectorsOnlyZeros /= float(size)
    print "Fraction of zeros: ", fractionOfZeros
    print "Fraction of input vectors with only zeros: ", fractionOfInputVectorsOnlyZeros
    print
    
    meanNeighbours /= float(size)
    print "Mean number of neighbours: ", meanNeighbours
    print
    
    print "min(Rjk) ", rjkMin
    print "max(Rjk) ", rjkMax
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
        print "Normalizing input..."
    
    if shiftMeanFlag:
        # shift inputs so that average is zero
        inputData  -= np.mean(inputData, axis=0)
        # shift outputs so that average is zero
        outputData -= np.mean(outputData, axis=0)
        print "Shifting input mean..."
    
    # scale the covariance
    if scaleCovarianceFlag:
        C = np.sum(inputData**2, axis=0) / float(size)
        print C
        print "Scaling input covariance..."
    
    if decorrelateFlag:     
        # decorrelate data
        cov = np.dot(inputData.T, inputData) / inputData.shape[0] # get the data covariance matrix
        U,S,V = np.linalg.svd(cov)
        inputData = np.dot(inputData, U) # decorrelate the data
        C = np.sum(inputData**2, axis=0) / float(size)
        print C
        print "Decorrelating input..."
          
    """print "New input data:"
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
    print"""  
    
    return inputData, outputData
    
    
    
normalizeFlag       = False    
shiftMeanFlag       = False
scaleCovarianceFlag = False
decorrelateFlag     = False


if __name__ == '__main__':
    pass
    
    
    