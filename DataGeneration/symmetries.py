import numpy as np
import tensorflow as tf
import sys

def cutoffFunction(R, Rc, cut=False):   
    
    value = 0.5 * (np.cos(np.pi*R / Rc) + 1)

    # set elements above cutoff to zero so they dont contribute to sum
    if cut:
        if isinstance(R, np.ndarray):
            value[np.where(R > Rc)[0]] = 0
        else:
            if R > Rc:
                value = 0
        
    return value
    
 
    
def G1(Rij, Rc):
    
    return np.sum(cutoffFunction(Rij, Rc))
    
    
def G2(Rij, eta, Rc, Rs):
    
    return np.sum( np.exp(-eta*(Rij - Rs)**2) * cutoffFunction(Rij, Rc) )
    
    
def G3(Rij, Rc, kappa):
    
    return np.sum( np.cos(kappa*Rij) * cutoffFunction(Rij, Rc))
    
    
def G4(Rij, Rik, Rjk, cosTheta, eta, Rc, zeta, Lambda):
    
    return 2**(1-zeta) * np.sum( (1 + Lambda*cosTheta)**zeta * \
           np.exp( -eta*(Rij**2 + Rik**2 + Rjk**2) ) * \
           cutoffFunction(Rij, Rc) * cutoffFunction(Rik, Rc) * cutoffFunction(Rjk, Rc, cut=True) )
           
           
def G5(Rij, Rik, cosTheta, eta, Rc, zeta, Lambda):
    
    return 2**(1-zeta) * np.sum( (1 + Lambda*cosTheta)**zeta* \
           np.exp( -eta*(Rij**2 + Rik**2) ) * \
           cutoffFunction(Rij, Rc) * cutoffFunction(Rik, Rc) )
           
           
           
def cutoffFunctionTF(R, Rc, cut=False):   
    
    value = 0.5 * (tf.cos(np.pi*R / Rc) + 1)

    # set elements above cutoff to zero so they dont contribute to sum
    """if cut:
        if isinstance(R, np.ndarray):
            value[np.where(R > Rc)[0]] = 0
        else:
            if R > Rc:
                value = 0"""
        
    return value
    
    
def G2TF(xij, yij, zij, eta, Rc, Rs):
    
    Rij = tf.sqrt(xij*xij + yij*yij + zij*zij)
    
    return tf.reduce_sum( tf.exp(-eta*(Rij - Rs)**2) * cutoffFunctionTF(Rij, Rc) )
    
    
def G4TF(xij, yij, zij, xik, yik, zik, eta, Rc, zeta, Lambda):
       
    Rij2 = xij*xij + yij*yij + zij*zij
    Rij = tf.sqrt(Rij2)
    
    Rik2 = xik*xik + yik*yik + zik*zik
    Rik = tf.sqrt(Rik2)
    
    cosTheta = (xij*xik + yij*yik + zij*zik) / (Rij*Rik)

    xjk = xij - xik
    yjk = yij - yik
    zjk = zij - zik
    Rjk = tf.sqrt(xjk*xjk + yjk*yjk + zjk*zjk)

    
    return 2**(1-zeta) * tf.reduce_sum( (1 + Lambda*cosTheta)**zeta * \
           tf.exp( -eta*(Rij2 + Rik2 + Rjk*Rjk) ) * \
           cutoffFunctionTF(Rij, Rc) * cutoffFunctionTF(Rik, Rc) * cutoffFunctionTF(Rjk, Rc, cut=True) )
           
        
def G5TF(xij, yij, zij, xik, yik, zik, eta, Rc, zeta, Lambda):
       
    Rij2 = xij*xij + yij*yij + zij*zij
    Rij = tf.sqrt(Rij2)
    
    Rik2 = xik*xik + yik*yik + zik*zik
    Rik = tf.sqrt(Rik2)
    
    cosTheta = (xij*xik + yij*yik + zij*zik) / (Rij*Rik)
   
    return 2**(1-zeta) * tf.reduce_sum( (1 + Lambda*cosTheta)**zeta * \
           tf.exp( -eta*(Rij2 + Rik2) ) * \
           cutoffFunctionTF(Rij, Rc) * cutoffFunctionTF(Rik, Rc) )
           
           
def dfcdr(R, Rc):
    
    return -0.5*(np.pi/Rc) * np.sin((np.pi*R) / Rc)
          
          
def dG2dr(xij, yij, zij, Rij, eta, Rc, Rs):
    
    dr = np.exp(-eta*(Rij - Rs)**2) * (2*eta*(Rs - Rij)*cutoffFunction(Rij, Rc) + dfcdr(Rij, Rc))
    fpair = -dr/Rij
    
    dij = []
    dij.append(fpair*xij)
    dij.append(fpair*yij)
    dij.append(fpair*zij)
    
    return dij
    
           
           
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
                inputData[i,symmFuncNumber] += G1(rij, s[0])              
            else:
                inputData[i,symmFuncNumber] += G2(rij, s[0], s[1], s[2])
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
    
    
def calculateForces(x, y, z, r, parameters, sampleDir, dEdG):
    
    print
    print "Computing forces..."
    
    size = len(x)
    
    outName = sampleDir + "/forces.txt"
    
    with open(outName, 'w') as outfile:
        
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
            f2x = np.zeros(numberOfNeighbours)
            f2y = np.zeros(numberOfNeighbours)
            f2z = np.zeros(numberOfNeighbours)
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
                symmFuncNumber = 0
                for s in parameters:
                    if len(s) == 3:
                        # compute derivative of G2 w.r.t. (x,y,z) of all neighbours simultaneously
                        dij = dG2dr(xij, yij, zij, rij, s[0], s[1], s[2]) 
                        f2x[j] += dEdG[i][symmFuncNumber]*dij[0]
                        f2y[j] += dEdG[i][symmFuncNumber]*dij[1]
                        f2z[j] += dEdG[i][symmFuncNumber]*dij[2]
                    else:
                        """if symmFuncType == 'G4':
                            inputData[i,symmFuncNumber] += G4(rij, rik, rjk, cosTheta, \
                                                              s[0], s[1], s[2], s[3])
                        else:
                            inputData[i,symmFuncNumber] += G5(rij, rik, cosTheta, \
                                                              s[0], s[1], s[2], s[3])"""
                        
                    symmFuncNumber += 1
                    
                outfile.write('%g %g %g ' % (f2x[j], f2y[j], f2z[j]))
            
            outfile.write('\n')
            
            # show progress
            sys.stdout.write("\r%2d %% complete" % ((float(i)/size)*100))
            sys.stdout.flush()
            
            # G2
            """if len(s) == 3:           

                for i in xrange(size):
                    # Rijs[i]: 1d array: [rij1, rij2, ...]
                    # drijs[i]: [[xij1, xij2, ... ], [yij1, yij2, ... ], [zij1, zij2, ... ]]
                    dij = np.sum( dG2dr(Rijs[i], drijs[i], s[0], s[1], s[2]) )  
                    diffj2x[i] = np.sum(dij[0])
                    diffj2y[i] = np.sum(dij[1])
                    diffj2z[i] = np.sum(dij[2])
            
            # G4
            else:
                # differentiate G4 w.r.t. all coordinates
                # need dG4/dxij and dG4/dxik for all j and k
                for i in xrange(size):
                    numberOfNeighbours = len(Rijs[i])
                    sumjx = 0; sumjy = 0; sumjz = 0;
                    sumkx = 0; sumky = 0; sumkz = 0;
                    for j in xrange(numberOfNeighbours):
                        # Rijs[i][j]: a number
                        # drijs[i][j]: 1d array [xij, yij, zij]
                        # Riks[i][j], Rjks[i][j], cosThetas[i][j]: 1d arrays
                        # driks[i][j], drjks[i][j] (if numberOfNeighbours = 4):
                        # list of 1d arrays[[xik2, xik3, xik4], [yik2, yik3, yik4], [zik2, zik3, zik4]]
                        dij, dik = dG4dr(Rijs[i][j], Riks[i][j], Rjks[i][j], cosThetas[i][j], \
                                         drijs[i][j], driks[i][j], drjks[i][j], \
                                         s[0], s[1], s[2], s[3])
                        sumjx += np.sum(dij[0])
                        sumjy += np.sum(dij[1])
                        sumjz += np.sum(dij[2])
                        sumkx += np.sum(dik[0])
                        sumky += np.sum(dik[1])
                        sumkz += np.sum(dik[2])
                    
                    diffj3x[i] = sumjx
                    diffj3y[i] = sumjy
                    diffj3z[i] = sumjz
                    diffk3x[i] = sumkx
                    diffk3y[i] = sumky
                    diffk3z[i] = sumkz"""
      
      
           
def applyThreeBodySymmetry(x, y, z, r, parameters, symmFuncType, function=None, E=None, forces=False,
                           sampleDir=''):
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
        
    # train only on parts of configs to test
    """indicies = np.arange(5)
    N = 5
    x = x[:3]
    y = y[:3]
    z = z[:3]
    r = r[:3]"""
    
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
            
            # all k != i,j OR I > J ??? REMEMBER TO CHANGE WHEN NEEDED
            k = np.arange(len(ri[:])) > j  
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
                    inputData[i,symmFuncNumber] += G2(rij, s[0], s[1], s[2])
                else:
                    if symmFuncType == 'G4':
                        inputData[i,symmFuncNumber] += G4(rij, rik, rjk, cosTheta, \
                                                          s[0], s[1], s[2], s[3])
                    else:
                        inputData[i,symmFuncNumber] += G5(rij, rik, cosTheta, \
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
        
    """if sampleDir:
        print 
        print "Writing symmetrized input data to file"
        with open(sampleDir + 'symmetryBehler.txt', 'w') as outfile:
            for vector in inputData:
                for symmValue in vector:
                    outfile.write('%g ' % symmValue)
                outfile.write('\n')"""
              
    # test where my SW-potential is equivalent with lammps SW-potential
    """Etmp = np.array(E[:20][:])
    outtmp = outputData[:20,:]
    print "Lammps:"
    print Etmp
    print 
    print "MySW:"
    print outtmp
    print 
    print outtmp - Etmp
    print 
    print outtmp/Etmp"""
    
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
    
    
    