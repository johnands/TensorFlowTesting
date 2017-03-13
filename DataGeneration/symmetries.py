import numpy as np
import sys

def cutoffFunction(rVector, Rc, cut=False):   
    
    value = 0.5 * (np.cos(np.pi*rVector / Rc) + 1)

    # set elements above cutoff to zero so they dont contribute to sum
    if cut:
        value[np.where(rVector > Rc)[0]] = 0
        
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
           
           
           
def dfcdr(rVector, Rc):
    
    return -0.5*(np.pi/Rc) * np.sin((np.pi*rVector) / Rc)
           
           
def dG2dr(Rij, eta, Rc, Rs):
    
    return np.exp(-eta*(Rij - Rs)**2) * (2*eta*(Rs - Rij)*cutoffFunction(Rij, Rc) + dfcdr(Rij, Rc))
    
    
def dG4dr(Rij, Rik, Rjk, cosTheta, xij, xik, yij, yik, zij, zik, eta, Rc, zeta, Lambda):
    
    F1 = 2**(1-zeta) * (1 + Lambda*cosTheta)**zeta
    F2 = np.exp( -eta*(Rij**2 + Rik**2 + Rjk**2) )
    
    F3 = cutoffFunction(Rij, Rc) * cutoffFunction(Rik, Rc) * cutoffFunction(Rjk, Rc, cut=True)
    
    dF1dcosTheta = 2**(1-zeta) * Lambda*zeta*(1 + Lambda*cosTheta)**(zeta-1)
    dF2dr = -2*eta*F2
    dF3drij = dfcdr(Rij, Rc) * cutoffFunction(Rik, Rc) * cutoffFunction(Rjk, Rc, cut=True)
    dF3drik = cutoffFunction(Rij, Rc) * dfcdr(Rik, Rc) * cutoffFunction(Rjk, Rc, cut=True)
    
    term1 = dF1dcosTheta * F2 * F3
    term2 = F1 * dF2dr * F3
    term3ij = F1 * F2 * dF3drij
    term3ik = F1 * F2 * dF3drik
    
    Rijinv = 1.0 / Rij
    Rikinv = 1.0 / Rik
    cosRijinv2 = cosTheta*Rijinv*Rijinv
    cosRikinv2 = cosTheta*Rikinv*Rikinv
    RijRikinv = 1.0 / (Rij*Rik)
    
    dij = []
    dik = []

    dij.append( np.sum( xij*(cosRijinv2*term1 - term2 - Rijinv*term3ij) - xik*(RijRikinv*term1) ) ) 
    dij.append( np.sum( yij*(cosRijinv2*term1 - term2 - Rijinv*term3ij) - yik*(RijRikinv*term1) ) )
    dij.append( np.sum( zij*(cosRijinv2*term1 - term2 - Rijinv*term3ij) - zik*(RijRikinv*term1) ) )
    
    dik.append(xik*(cosRikinv2*term1 - term2 - Rikinv*term3ik) - xij*(RijRikinv*term1))
    dik.append(yik*(cosRikinv2*term1 - term2 - Rikinv*term3ik) - yij*(RijRikinv*term1))
    dik.append(zik*(cosRikinv2*term1 - term2 - Rikinv*term3ik) - zij*(RijRikinv*term1))
    
    return dij, dik 
    
    
           
           
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
      
      
           
def applyThreeBodySymmetry(x, y, z, r, parameters, function=None, E=None):
    """
    Transform input coordinates with 2- and 3-body symmetry functions
    Input coordinates can be random or sampled from lammps
    Output data can be supplied with an array E or be generated
    using the optional function argument
    """
    
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
        #print "Energy is generated with user-supplied function"
    
    # loop through each data vector, i.e. each atomic environment
    fractionOfNonZeros = 0.0
    fractionOfInputVectorsOnlyZeros = 0.0
    meanNeighbours = 0.0
    rjkMin = 100.0
    rjkMax = 0.0
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
            
            rjk = np.sqrt( rij**2 + rik**2 - 2*rij*rik*cosTheta )
            
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
                    inputData[i,symmFuncNumber] += G4(rij, rik, rjk, cosTheta, \
                                                      s[0], s[1], s[2], s[3])
                symmFuncNumber += 1

            # calculate energy with supplied 3-body function or with
            if function != None:
                outputData[i,0] += function(rij, rik, cosTheta)                   
            
        # shuffle input vector
        #np.random.shuffle(inputData[i,:])
        #print outputData[i,0]
        
        # count zeros
        fractionOfNonZeros += np.count_nonzero(inputData[i,:]) / float(numberOfSymmFunc)
        if not np.any(inputData[i,:]):
            fractionOfInputVectorsOnlyZeros += 1
            print i
            
        # show progress
        sys.stdout.write("\r%2d %% complete" % ((float(i)/size)*100))
        sys.stdout.flush()
        
        
    # test where my SW-potential is equivalent with lammps SW-potential
    """Etmp = np.array(E[:20][:])*2
    outtmp = outputData[:20,:]
    print "Lammps:"
    print Etmp
    print 
    print "MySW:"
    print outtmp
    print 
    print outtmp - Etmp
    print 
    print outtmp/Etmp
    exit(1)"""
    
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
    
    
    