# make dataset which consists of random integers 0-9 in pairs
# for example training of ANN

import numpy as np
import symmetryFunctions
np.random.seed(1)

def compareIntegers(N, a=0, b=10):
    """
    Make training and test data which consists of random integers [a,b] in pairs
    for training neural network to identify which number is largest in each pair
    """
    
    # create a vector [N,2] of N pairs of random integers between a and b
    # this is the training data
    x_train = np.random.randint(a, b, size=(N,2))
    x_test  = np.random.randint(a, b, size=(N,2))
    
    # make test data, contains zeros and ones
    y_train = np.zeros((N,3))
    y_test  = np.zeros((N,3))
      
    # turn into boolean matrix  
    y_train[:,0] = x_train[:,0] > x_train[:,1]   
    y_train[:,2] = x_train[:,0] < x_train[:,1]
    y_test[:,0] = x_test[:,0] > x_test[:,1]   
    y_test[:,2] = x_test[:,0] < x_test[:,1]
    
    # equal
    y_train[:,1] = x_train[:,0] == x_train[:,1]
    y_test[:,1] = x_test[:,0] == x_test[:,1]
    
    # convert to floats: ones and zeros
    y_train *= 1
    y_test *= 1
    
    return x_train, y_train, x_test, y_test
    
    
def functionData(function, trainSize, testSize, a=0.8, b=2.5):
    """
    Create random numbers as input for neural network
    to approximate any continous function
    """
    
    x_train = np.random.uniform(a, b, trainSize)
    x_train = x_train.reshape([trainSize,1])
    y_train = function(x_train)
    
    x_test  = np.random.uniform(a, b, testSize)
    x_test  = x_test.reshape([testSize,1])
    y_test  = function(x_test)
    
    return x_train, y_train, x_test, y_test
    
    
def neighbourData(function, size, a, b, inputs, outputs=1):
    """
    Create random distances [a,b] for N (inputs) neighbouring atoms
    and make output set based on these random points
    The output node is the sum of the energy of all neighbours
    """
    
    dimension = (size, inputs)
    inputData = np.random.uniform(a, b, dimension)
    outputData = np.sum(function(inputData), axis=1)
    outputData = outputData.reshape([size,outputs])
    
    return inputData, outputData
    
def neighbourDataVarying(function, size, a, b, minNeigh, maxNeigh, outputs=1):
    """
    Create random distances [a,b] for varying number of neighbours
    and make output set based on these random points
    The output node is the sum of the energy of all neighbours
    """
    
    inputData = []
    outputData = []
    numberOfNeighbours = np.random.randint(minNeigh, maxNeigh, size=size)
    for i in xrange(size):
        inputData.append(np.random.uniform(a, b, numberOfNeighbours[i]))
        outputData.append( sum(function(inputData[i])) )
        
    return inputData, outputData
        
    
    
def neighbourEnergyAndForceData(function, functionDerivative, size, \
                                inputs, outputs=4, a=0.8, b=2.5):
    """
    Create random coordinates (x,y,z) on [a,b] for N (inputs) neighbouring atoms
    and make output set which consist of total energy and total force
    in each direction. The NN will thus yield the total energy and force
    from N surrounding atoms
    """
    
    coordDimension = (size, inputs, 3)
    
    # make random coordinates (x,y,z)
    coordinates = np.random.uniform(0.0, 2.5, coordDimension) * np.random.choice([-1,1], coordDimension)
    
    # make training set distances
    inputData = coordinates[:,:,0]**2 + coordinates[:,:,1]**2 + coordinates[:,:,2]**2
    inputData = np.sqrt(inputData)
    
    # delete all input vectors which have at least one element below 0.8
    indicies = np.where(inputData < a)[0]
    indicies = np.unique(indicies)
    inputData = np.delete(inputData, indicies, axis=0)
    coordinates = np.delete(coordinates, indicies, axis=0)
    
    # adjust dimension of output after deleting rows        
    outputDimension = (inputData.shape[0], outputs)            
    outputData = np.zeros(outputDimension)
    
    # first element of yTrain is sum of energies of all neighbours
    outputData[:,0] = np.sum(function(inputData), axis=1)
    
    # 2,3,4 components are sum of Fx, Fy, Fz respectively for all neighbours
    inverseDistances = 1.0/inputData
    outputData[:,1] = np.sum(functionDerivative(inputData)*coordinates[:,:,0]*inverseDistances, axis=1)
    outputData[:,2] = np.sum(functionDerivative(inputData)*coordinates[:,:,1]*inverseDistances, axis=1)
    outputData[:,3] = np.sum(functionDerivative(inputData)*coordinates[:,:,2]*inverseDistances, axis=1)     
    
    return inputData, outputData
    

def energyAndForceCoordinates(function, functionDerivative, size, \
                             neighbours, outputs=4, a=0.8, b=2.5):
    """
    Train with both potential and forces
    Input: (size, 4*neighbours) 
            [[x1,  y1,  z1,  r1, x2, y2, z2, r2, ....],
             [x1', y1', z1', r1', x2', ....          ],
             ...........                              ]
              
    Output: (size, 4)
    [[totalFx, totalFy, totalFz, totalEp],
     [......                            ]]
    """

    # generate input data
    inputData = np.zeros((size,neighbours,4))
    xyz     = np.zeros((size,3))
    for i in xrange(neighbours): # fill cube slice for each neighbor
        inputData[:,i,3] = np.random.uniform(0.8, 2.5, size) # this is R
        r2             = inputData[:,i,3]**2
        xyz[:,0]       = np.random.uniform(0, r2, size)
        xyz[:,1]       = np.random.uniform(0, r2-xyz[:,0], size)
        xyz[:,2]       = r2 - xyz[:,0] - xyz[:,1]
        for row in xrange(size):
            np.random.shuffle(xyz[row,:]) # this shuffles in-place (so no copying)
        inputData[:,i,0] = np.sqrt(xyz[:,0]) * np.random.choice([-1,1],size)
        inputData[:,i,1] = np.sqrt(xyz[:,1]) * np.random.choice([-1,1],size)
        inputData[:,i,2] = np.sqrt(xyz[:,2]) * np.random.choice([-1,1],size)
           
    # generate output data
    size    = inputData.shape[0]
    outputData = np.zeros((size, 4)) # 4: Fx, Fy, Fz and Ep
    r       = inputData[:,:,3]
      
    # sum up contribution from all neighbors:
    outputData[:,0] = np.sum( (functionDerivative(r) * inputData[:,:,0] / r), axis=1) # Fx
    outputData[:,1] = np.sum( (functionDerivative(r) * inputData[:,:,1] / r), axis=1) # Fy
    outputData[:,2] = np.sum( (functionDerivative(r) * inputData[:,:,2] / r), axis=1) # Fz
    outputData[:,3] = np.sum( function(r), axis=1 ) # Ep_tot
    
    return inputData.reshape(size, neighbours*4), outputData


def radialSymmetryData(function, size, \
                       neighbours, numberOfSymmFunc, symmFuncType, \
                       a, b, outputs=1):
        
    # generate train and test distances, which are vectors with dimension (trainSize, inputs)    
    # the output data is the same as before: a sum of LJ energies for all neighbours 
    # inputs defined in above function neighbourData is now number of neighbours               
    #inputTemp, outputData = neighbourData(function, size, a, b, neighbours)
    inputTemp, outputData = neighbourDataVarying(function, size, a, b, 30, 70)
    outputData = np.array(outputData)
    
    # send each distance input vector to a symmetry function which returns a single number
    # for each vector of distances
    # number of inputs to NN is now number of symmetry functions
    inputData = np.zeros((size,numberOfSymmFunc))
    print inputData.shape
    
    parameters = []
    if symmFuncType == 'G1':
        a += 0.2
        cutoffs = np.linspace(a, b, numberOfSymmFunc)
        parameters = cutoffs
        for i in xrange(size):
            # find value of each symmetry function for this r vector
            for j in xrange(numberOfSymmFunc):
                # remove distances above cutoff, they contribute zero to sum
                rVector = np.array(inputTemp[i][:])
                rVector = rVector[i,np.where(rVector <= cutoffs[j])[0]]
                inputData[i,j] = symmetryFunctions.G1(rVector, cutoffs[j])
            
    else:  
        
        # parameters
        cutoffs = [b]
        widths = [0.001, 0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.1, 0.3, 0.7]
        centers = [0.0, 3.1, 4.5, 5.2, 5.9, 6.8, 7.8]
        
        # collect all parameters in nested list
        for width in widths:
            for cutoff in cutoffs:
                for center in centers:   
                    parameters.append([width, cutoff, center])
                    
        # transform input data
        fractionOfNonZeros = 0.0
        fractionOfInputVectorsOnlyZeros = 0.0
        for i in xrange(size):
            j = 0
            # find value of each symmetry function for this r vector
            for width in widths:
                for cutoff in cutoffs:
                    for center in centers:                    
                        # remove distances above cutoff, they contribute zero to sum
                        rVector = np.array(inputTemp[i][:])
                        inputData[i,j] = symmetryFunctions.G2(rVector, cutoff, width, center)
                        j += 1
                    
            # count zeros
            fractionOfNonZeros += np.count_nonzero(inputData[i,:]) / float(numberOfSymmFunc)
            if not np.any(inputData[i,:]):
                fractionOfInputVectorsOnlyZeros += 1
        
    if not np.all(inputData):
        print 'zeros are present'
        
    fractionOfZeros = 1 - fractionOfNonZeros / float(size)
    fractionOfInputVectorsOnlyZeros /= float(size)
    print "Fraction of zeros: ", fractionOfZeros
    print "Fraction of input vectors with only zeros: ", fractionOfInputVectorsOnlyZeros
    
    outputData = outputData.resize([size, numberOfSymmFunc])
    inputData = inputData.resize([size, outputs])
    
    print inputData
    print outputData
    
    return inputData, outputData, parameters
    
    
def angularSymmetryData(function, size, \
                        neighbours, numberOfSymmFunc, symmFuncType, \
                        low, high, outputs=1):

    # generate input data
    inputTemp = np.zeros((size,neighbours,4))
    xyz     = np.zeros((size,3))
    for i in xrange(neighbours): # fill cube slice for each neighbor
        inputTemp[:,i,3] = np.random.uniform(low, high, size) # this is R
        r2             = inputTemp[:,i,3]**2
        xyz[:,0]       = np.random.uniform(0, r2, size)
        xyz[:,1]       = np.random.uniform(0, r2-xyz[:,0], size)
        xyz[:,2]       = r2 - xyz[:,0] - xyz[:,1]
        for row in xrange(size):
            np.random.shuffle(xyz[row,:]) # this shuffles in-place (so no copying)
        inputTemp[:,i,0] = np.sqrt(xyz[:,0]) * np.random.choice([-1,1],size)
        inputTemp[:,i,1] = np.sqrt(xyz[:,1]) * np.random.choice([-1,1],size)
        inputTemp[:,i,2] = np.sqrt(xyz[:,2]) * np.random.choice([-1,1],size)
    
    # pick out slices
    x = inputTemp[:,:,0]
    y = inputTemp[:,:,1]
    z = inputTemp[:,:,2]
    r = inputTemp[:,:,3]

    outputData = np.zeros((size, outputs))
    
    # generate symmetry function input data
    inputData = np.zeros((size,numberOfSymmFunc))
    
    # parameters
    widths = [0.1, 0.21, 0.32, 0.425, 0.526]    
    cutoffs = [high]
    thetaRange = [1, 2, 4]
    inversions = [-1.0, 1.0]
    
    # collect all parameters in nested list
    parameters = []
    for width in widths:
        for cutoff in cutoffs:
            for zeta in thetaRange:                                 
                for inversion in inversions:
                    parameters.append(width, cutoff, zeta, inversion)
    
    #counter = 0
    # loop through each r vector, i.e. each atomic environment
    for i in xrange(size):
        """if np.size( np.where(rjk[i,:] > cutoff)[0] ) == neighbours:
            counter += 1"""       
        # nested sum over all neighbours k for each neighbour j
        # this loop takes care of both 2-body and 3-body configs
        for j in xrange(neighbours):

            # pick coordinates and r of neighbour j
            rij = r[i,j]
            xij = x[i,j]; yij = y[i,j]; zij = z[i,j]
            
            # pick all neighbours k 
            indicies = np.arange(len(r[i,:])) != j
            rik = r[i,indicies] 
            xik = x[i,indicies]; yik = y[i,indicies]; zik = z[i,indicies]
            
            # compute angle and rjk
            theta = np.arccos( (xij*xik + yij*yik + zij*zik) / (rij*rik) )
            rjk = np.sqrt( rij**2 + rik**2 - 2*rij*rik*np.cos(theta) )
            
            # find value of each symmetry function for this triplet
            symmFuncNumber = 0
            for width in widths:
                for cutoff in cutoffs:
                    for zeta in thetaRange:                                 
                        for inversion in inversions:
                            # find symmetry function value for triplets (i,j,k) for all k
                            inputData[i,symmFuncNumber] += symmetryFunctions.G4(rij, rik, rjk, theta, \
                                                                                zeta, width, cutoff, inversion)
                            symmFuncNumber += 1
                                           
            # 3-body, rik and theta are vectors
            outputData[i,0] += np.sum(function(rij, rik, theta))
    
    # check if any input data is zero, should prevent this
    if not np.all(inputData):
        print 'zeros are present'
        
    maxValue = np.max(inputData)
    minValue = np.min(inputData)
    print maxValue
    print minValue
    print np.mean(inputData)
    
    # normalize to [-1,1]
    inputData = 2 * (inputData - minValue) / (maxValue - minValue) - 1 
    print "New max: ", np.max(inputData)
    print "New min: ", np.min(inputData)
    print "New ave: ", np.mean(inputData)   
    
    print "Ouput data:"
    print np.max(outputData)
    print np.min(outputData)
    print np.mean(outputData)
    
    # normalize output data
    maxOutput = np.max(outputData)
    minOutput = np.min(outputData)
    outputData = 2 * (outputData - minOutput) / (maxOutput - minOutput) - 1 
    
    print "New output:"
    print np.max(outputData)
    print np.min(outputData)
    print np.mean(outputData)
    
    return inputData, outputData, parameters
    
    


if __name__ == '__main__':
    #x_train, y_train, x_test, y_test = compareIntegers(10, 5, 15)
    function = lambda s : 1.0/s**12 - 1.0/s**6
    functionDerivative = lambda s : (6*s**6 - 12) / s**13
    #x_train, y_train, x_test, y_test = functionData(function, 100, 100)
    xTrain, yTrain, xTest, yTest = neighbourEnergyAndForceData(function, functionDerivative, \
                                                 1000, 10, 5)
    print xTrain
    """print yTrain.shape
    print xTest.shape
    print yTest.shape"""
    
