# make dataset which consists of random integers 0-9 in pairs
# for example training of ANN

import numpy as np
import symmetries
from time import clock as timer
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
    
    
    
def neighbourTwoBodyEnergyAndForce1(function, functionDerivative, size, \
                                    inputs, outputs=4, a=0.8, b=2.5):
    """
    Create random coordinates (x,y,z) on [a,b] for N (inputs) neighbouring atoms
    and make output set which consist of total energy and total force
    in each direction. The NN will thus yield the total energy and force
    from N surrounding atoms
    Input: r
    Output: (E, Fx, Fy, Fz)
    Doesn't work - too little information for the NN, must input (x,y,z) too
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
    
    
def neighbourTwoBodyEnergyAndForce2(function, functionDerivative, size, \
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
    
    inputData = neighbourCoordinatesInput(size, a, b, neighbours)
           
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
    
    
    
    
    
    

##### relevant functions #####   
    
def neighbourCoordinatesInput(size, a, b, neighbours):
    
    # generate input data
    inputData = np.zeros((size,neighbours,4))
    xyz     = np.zeros((size,3))
    for i in xrange(neighbours): # fill cube slice for each neighbor
        r                = np.random.uniform(a, b, size)
        r2               = r**2
        xyz[:,0]         = np.random.uniform(0, r2, size)
        xyz[:,1]         = np.random.uniform(0, r2-xyz[:,0], size)
        xyz[:,2]         = r2 - xyz[:,0] - xyz[:,1]
        for row in xrange(size):
            np.random.shuffle(xyz[row,:]) # this shuffles in-place (so no copying)
        inputData[:,i,0] = np.sqrt(xyz[:,0]) * np.random.choice([-1,1],size)
        inputData[:,i,1] = np.sqrt(xyz[:,1]) * np.random.choice([-1,1],size)
        inputData[:,i,2] = np.sqrt(xyz[:,2]) * np.random.choice([-1,1],size)
        inputData[:,i,3] = r2
        
    return inputData
    

def varyingNeighbourCoordinatesInput(size, a, b, minNeigh, maxNeigh):
    """
    Create random distances [a,b] and coordinates (x,y,z) for varying number of neighbours
    and make output set based on these random points
    The output node is the sum of the energy of all neighbours
    """
    
    x = []; y = []; z = []; r = []
    numberOfNeighbours = np.random.randint(minNeigh, maxNeigh, size=size)
    for i in xrange(size):
        N = numberOfNeighbours[i]
        ri = np.random.uniform(a, b, N)
        r2 = ri**2
        xyz = np.zeros((3,N))
        xyz[0] = np.random.uniform(0, r2, N)
        xyz[1] = np.random.uniform(0, r2-xyz[0], N)
        xyz[2] = r2 - xyz[0] - xyz[1]
        
        # this shuffles in-place (so no copying)
        # SHOULD NOT SHUFFLE:  THEN (xi, yi, zi) do not correspond with (ri) anymore
        #for dim in xrange(3):
        #    np.random.shuffle(xyz[dim]) 
            
        xyz[0] = np.sqrt(xyz[0]) * np.random.choice([-1,1], N)
        xyz[1] = np.sqrt(xyz[1]) * np.random.choice([-1,1], N)
        xyz[2] = np.sqrt(xyz[2]) * np.random.choice([-1,1], N)
        
        x.append( xyz[0].tolist() )
        y.append( xyz[1].tolist() )
        z.append( xyz[2].tolist() )
        r.append( r2.tolist() )
             
    return x, y, z, r



def twoBodyEnergy(function, trainSize, testSize, a=0.8, b=2.5):
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
 
   
    
def neighbourTwoBodyEnergy(function, size, a, b, inputs, outputs=1):
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
   
   
    
def varyingNeighbourTwoBodyEnergy(function, size, a, b, minNeigh, maxNeigh):
    """
    Create random distances [a,b] for varying number of neighbours
    and make output set based on these random points
    The output node is the sum of the energy of all neighbours
    """
    
    # input data have varying number of rows, must be a list
    # and converted to array later
    inputData = []
    outputData = np.zeros((size,1))
    numberOfNeighbours = np.random.randint(minNeigh, maxNeigh, size=size)
    for i in xrange(size):
        Rij = np.random.uniform(a, b, numberOfNeighbours[i])
        outputData[i,0] = np.sum( function(Rij) )
        inputData.append(Rij.tolist())
        
    return inputData, outputData
        
        

def neighbourTwoBodySymmetry(function, size, \
                             neighbours, numberOfSymmFunc, symmFuncType, \
                             a, b, outputs=1, varyingNeigh=True,
                             minNeigh=30, maxNeigh=70):      
      
    # generate neighbours lists of varying sizes or not
    # each list contains only distances to neighbours
    if varyingNeigh:
        # input: list, output: array
        inputTemp, outputData = varyingNeighbourTwoBodyEnergy(function, size, a, b, minNeigh, maxNeigh)
    else:
        # input: array, output: array
        inputTemp, outputData = neighbourTwoBodyEnergy(function, size, a, b, neighbours)
  
    # symmetry function parameters
    cutoffs = [b]
    widths = [0.001, 0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.1, 0.3, 0.7]
    centers = [0.0, 3.1, 4.5, 5.2, 5.9, 6.8, 7.8]
        
    # collect all parameters in nested list
    parameters = []
    for width in widths:
        for cutoff in cutoffs:
            for center in centers:   
                parameters.append([width, cutoff, center])
                
    # apply symmetry transformation to input data
    inputData = symmetries.applyTwoBodySymmetry(inputTemp, parameters)
                
    return inputData, outputData, parameters
    
    
    
def neighbourThreeBodySymmetry(function, size, \
                               neighbours, numberOfSymmFunc, symmFuncType, \
                               a, b, outputs=1, varyingNeigh=True,
                               minNeigh=4, maxNeigh=15):
    """
    Produce 3-body symmetry-transformed random input data
    The neighbours lists can have varying number of neighbours (varyingNeigh == True) or not
    """

    # generate random coordinates  
    if varyingNeigh:
        # x, y, z, r: lists
        x, y, z, r = varyingNeighbourCoordinatesInput(size, a, b, minNeigh, maxNeigh)
    else:
        # x, y, z, r: arrays
        inputTemp = neighbourCoordinatesInput(size, a, b, neighbours)
        x = inputTemp[:,:,0]
        y = inputTemp[:,:,1]
        z = inputTemp[:,:,2]
        r = inputTemp[:,:,3]
        
    # NOTE: r is now r^2 because I want correspondence with lammps data,
    # where all r's are squared
    
    # parameters G2
    widthG2 = [0.001, 0.01, 0.1]
    cutoffG2 = [4.0]
    centerG2 = [0.0, 3.0]

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
                    
    print len(parameters)
    print parameters
    
    # apply symmetry transformation to input data and generate output data
    inputData, outputData = symmetries.applyThreeBodySymmetry(x, y, z, r, parameters, function=function)
    
    return inputData, outputData, parameters
    
    


if __name__ == '__main__':
    pass
