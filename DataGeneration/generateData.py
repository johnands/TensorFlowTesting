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
    
    
def functionData(function, N, a=0, b=1):
    """
    Create random numbers as input for neural network
    to approximate any continous function
    """
    
    x_train = np.random.uniform(a, b, N)
    x_train = x_train.reshape([N,1])
    y_train = function(x_train)
    
    x_test  = np.random.uniform(a, b, N)
    x_test  = x_test.reshape([N,1])
    y_test  = function(x_test)
    
    return x_train, y_train, x_test, y_test
    


if __name__ == '__main__':
    #x_train, y_train, x_test, y_test = compareIntegers(10, 5, 15)
    x_train, y_train, x_test, y_test = functionData(10)
    
