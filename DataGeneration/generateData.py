# make dataset which consists of random integers 0-9 in pairs
# for example training of ANN

import numpy as np

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
    np.random.seed(1)
    x_train = np.random.uniform(a, b, trainSize)
    x_train = x_train.reshape([trainSize,1])
    y_train = function(x_train)
    
    x_test  = np.random.uniform(a, b, testSize)
    x_test  = x_test.reshape([testSize,1])
    y_test  = function(x_test)
    
    return x_train, y_train, x_test, y_test
    
    
def neighbourData(function, trainSize, testSize, a=0.8, b=2.5, inputs=5, outputs=1):
    """
    Create random distances [a,b] for five neighbouring atoms
    and make test set based on these random points
    The output node is the sum of the energy of all neighbours
    """
    np.random.seed(1)
    
    # xTrain: shape(trainSize, neighbours)
    # yTrain: shape(trainSize, outputs)
    dimension = (trainSize, inputs)
    xTrain = np.random.uniform(a, b, dimension)
    yTrain = np.sum(function(xTrain), axis=1)
    yTrain.reshape([trainSize,outputs])
    
    dimension = (testSize, inputs)
    xTest = np.random.uniform(a, b, dimension)
    yTest = np.sum(function(xTest), axis=1)
    yTrain.reshape([testSize,outputs])
    
    return xTrain, yTrain, xTest, yTest
    
    


if __name__ == '__main__':
    #x_train, y_train, x_test, y_test = compareIntegers(10, 5, 15)
    function = lambda s : 1.0/s**12 - 1.0/s**6
    x_train, y_train, x_test, y_test = functionData(function, 100, 100)
    
