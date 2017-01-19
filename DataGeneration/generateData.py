# make dataset which consists of random integers 0-9 in pairs
# for example training of ANN

import numpy as np
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
    
    
def neighbourData(function, size, a=0.8, b=2.5, inputs=5, outputs=1):
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
    for i in range(neighbours): # fill cube slice for each neighbor
        inputData[:,i,3] = np.random.uniform(0.8, 2.5, size) # this is R
        r2             = inputData[:,i,3]**2
        xyz[:,0]       = np.random.uniform(0, r2, size)
        xyz[:,1]       = np.random.uniform(0, r2-xyz[:,0], size)
        xyz[:,2]       = r2 - xyz[:,0] - xyz[:,1]
        for row in range(size):
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
                       outputs=1, a=0.8, b=2.5):
        
    # generate train and test distances, which are vectors with dimension (trainSize, inputs)    
    # the output data is the same as before: a sum of LJ energies for all neighbours 
    # inputs defined in above function neighbourData is now number of neighbours               
    inputTemp, outputData = neighbourData(function, size, \
                                          a=a, b=b, inputs=neighbours, outputs=1)
    
    # send each distance input vector to a symmetry function which returns a single number
    # for each vector of distances
    # number of inputs to NN is now number of symmetry functions
    inputData = np.zeros((size,numberOfSymmFunc))
    
    if symmFuncType == 'G1':
        cutoff = np.linspace(0.9, 3, numberOfSymmFunc)
        #cutoff = [2.5]
        for i in xrange(size):
            # find value of each symmetry function for this r vector
            for j in xrange(numberOfSymmFunc):
                # remove distances above cutoff, they contribute zero to sum
                rVector = inputTemp[i,np.where(inputTemp[i,:] <= cutoff[j])[0]]
                inputData[i,j] = symmetryFunction1(rVector, cutoff[j])
                
    else:  
        cutoff = 2.5
        widths = np.linspace(0.9, 2.8, numberOfSymmFunc)
        center = 0.0
        for i in xrange(size):
            # find value of each symmetry function for this r vector
            for j in xrange(numberOfSymmFunc):
                # remove distances above cutoff, they contribute zero to sum
                rVector = inputTemp[i,:]
                inputData[i,j] = symmetryFunction2(rVector, cutoff, widths[j], center)
        
    if not np.all(inputData):
        print 'zeros are present'
        
    print inputData.shape
    
    return inputData, outputData
    
    
def angularSymmetryData(function, size, \
                        neighbours, numberOfSymmFunc, symmFuncType, \
                        outputs=1, a=0.8, b=2.5):

    # generate input data
    inputTemp = np.zeros((size,neighbours+1,4))
    xyz     = np.zeros((size,3))
    for i in range(neighbours+1): # fill cube slice for each neighbor
        inputTemp[:,i,3] = np.random.uniform(0.8, 2.5, size) # this is R
        r2             = inputTemp[:,i,3]**2
        xyz[:,0]       = np.random.uniform(0, r2, size)
        xyz[:,1]       = np.random.uniform(0, r2-xyz[:,0], size)
        xyz[:,2]       = r2 - xyz[:,0] - xyz[:,1]
        for row in range(size):
            np.random.shuffle(xyz[row,:]) # this shuffles in-place (so no copying)
        inputTemp[:,i,0] = np.sqrt(xyz[:,0]) * np.random.choice([-1,1],size)
        inputTemp[:,i,1] = np.sqrt(xyz[:,1]) * np.random.choice([-1,1],size)
        inputTemp[:,i,2] = np.sqrt(xyz[:,2]) * np.random.choice([-1,1],size)
    
    xij = inputTemp[:,:-1,0]; xik = inputTemp[:,1:,0]
    yij = inputTemp[:,:-1,1]; yik = inputTemp[:,1:,1]
    zij = inputTemp[:,:-1,2]; zik = inputTemp[:,1:,2]
    rij = inputTemp[:,:-1,3]; rik = inputTemp[:,1:,3]
    
    theta = np.arccos( (xij*xik + yij*yik + zij*zik) / (rij*rik) )
    
    outputData = np.sum(function(rij, rik, theta), axis=1)
    
    
    
def cutoffFunction(rVector, cutoff):
    
    return 0.5 * (np.cos(np.pi*rVector / cutoff) + 1)
    
    
def symmetryFunction1(rVector, cutoff):
    
    return np.sum(cutoffFunction(rVector, cutoff))
    
    
def symmetryFunction2(rVector, cutoff, width, center):
    
    return np.sum( np.exp(-width*(rVector - center)**2) * cutoffFunction(rVector, cutoff) )
    
    
def symmetryFunction3(Rij, Rik, Rjk, theta, thetaRange, width, Lambda):
    
    return 2**(1-thetaRange) * np.sum(1 + Lambda*np.cos(theta))**thetaRange * \
           np.exp(-width*(Rij**2 + Rik**2 + Rjk**2)) * \
           cutoffFunction(Rij) * cutoffFunction(Rik) * cutoffFunction(Rjk)
    


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
    
