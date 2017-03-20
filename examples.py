import regression
import numpy as np
import tensorflow as tf

def performanceTest(maxEpochs, maxLayers, maxNodes):
    
    # function to approximate
    function = lambda s : 1.0/s**12 - 1.0/s**6
    
    # approximate on [a,b]
    a = 0.9
    b = 1.6

    regress = regression.Regression(function, int(1e6), int(1e4), int(1e3))
    regress.generateData(a, b)
    
    # finding optimal value
    counter = 0
    for layers in range(1, maxLayers+1, 1):
        for nodes in range(layers, maxNodes+1, 1):
            start = timer()
            regress.constructNetwork(layers, nodes)
            regress.train(maxEpochs)
            end = timer()
            timeElapsed = end - start
            print "Layers: %2d, nodes: %2d, time = %10g" % (layers, nodes, timeElapsed)
            print
    
            if counter == 0:
                with open('Tests/timeElapsed', 'w') as outFile:
                    outStr = "Timing analysis"
                    outFile.write(outStr + '\n')
                    
            with open('Tests/timeElapsed', 'a') as outFile:
                outStr = "Layers: %2d, nodes: %2d, time = %10g" % (layers, nodes, timeElapsed)
                outFile.write(outStr + '\n')
            
            counter += 1
            
def testActivations(trainSize, batchSize, testSize, nLayers, nNodes, nEpochs, a=0.9, b=1.6):
    
    # function to approximate
    function = lambda s : 1.0/s**12 - 1.0/s**6
    
    # approximate on [a,b]
    a = 0.9
    b = 1.6

    regress = regression.Regression(function, int(1e6), int(1e4), int(1e3), 1, 1)
    regress.generateData(a, b)
    
    # test different activations
    activations = [tf.nn.relu, tf.nn.relu6, tf.nn.elu, tf.nn.sigmoid, tf.nn.tanh]
    counter = 0
    for act in activations:    
        regress.constructNetwork(nLayers, nNodes, activation=act, wInit='trunc_normal', bInit='trunc_normal')
        regress.train(nEpochs)
        counter += 1
        

def setUpLJPotential(units, shifted=False):
    
    ### metal units ###
    if units == 'metal':
        a = 3.2
        cutoff = 8.5125
        epsilon = 1.0
        sigma = 3.405
        
    ### LJ units ###
    else:
        a = 0.8
        cutoff = 2.5
        epsilon = 1.0
        sigma = 1.0
        
    if shifted:
        shiftedPotential = sigma**12/cutoff**12 - sigma**6/cutoff**6
        function = lambda s : 4*epsilon*(sigma**12/s**12 - sigma**6/s**6 - shiftedPotential)
        
    else:
        function = lambda s : 4*epsilon*(sigma**12/s**12 - sigma**6/s**6)
        
    return function, a, cutoff
        

 
def LennardJonesExample(trainSize, batchSize, testSize, nLayers, nNodes, nEpochs, 
                        units='metal'):
    """
    Train to reproduce shifted L-J potential to 
    verify implementation of network and backpropagation in the MD code
    This is a 1-dimensional example
    """

    function, a, cutoff = setUpLJPotential('metal')    
    
    regress = regression.Regression(function, trainSize, batchSize, testSize, 1, 1)
    regress.generateData(a, cutoff, 'twoBody')
    regress.constructNetwork(nLayers, nNodes, activation=tf.nn.sigmoid, \
                             wInit='normal', bInit='normal')
    regress.train(nEpochs)

    
def LennardJonesNeighbours(trainSize, batchSize, testSize, nLayers, nNodes, nEpochs, \
                           neighbours, outputs=1):
                               
    function, a, b = setUpLJPotential('LJ')

    regress = regression.Regression(function, trainSize, batchSize, testSize, neighbours, outputs)
    regress.generateData(a, b, 'neighbourTwoBody')
    regress.constructNetwork(nLayers, nNodes, activation=tf.nn.sigmoid, \
                             wInit='normal', bInit='normal')
    regress.train(nEpochs)
                               
                               
def LennardJonesNeighboursForce(trainSize, batchSize, testSize, nLayers, nNodes, nEpochs, \
                                neighbours, outputs=4, a=0.8, b=2.5):
    
    function, a, b = setUpLJPotential('LJ', shifted=True)
    functionDerivative = lambda t : 12.0/t**13 - 6.0/t**7
    inputs = neighbours*4
    regress = regression.Regression(function, trainSize, batchSize, testSize, inputs, outputs, \
                         functionDerivative)
    regress.generateData(a, b, 'neighbourTwoBody')
    regress.constructNetwork(nLayers, nNodes, activation=tf.nn.sigmoid, \
                             wInit='normal', bInit='normal')
    regress.train(nEpochs)
    

def LennardJonesSymmetryFunctions(trainSize, batchSize, testSize, nLayers, nNodes, nEpochs, \
                                  neighbours, numberOfSymmFunc, symmFuncType, outputs=1, \
                                  units='metal', shifted=False, varyingNeigh=True):
                                      
    function, a, b = setUpLJPotential('metal', shifted=shifted)
    
    regress = regression.Regression(function, trainSize, batchSize, testSize, numberOfSymmFunc, outputs)
    regress.generateData(a, b, 'twoBodySymmetry', neighbours=neighbours, numberOfSymmFunc=numberOfSymmFunc, 
                         symmFuncType='G2', varyingNeigh=varyingNeigh)
    regress.constructNetwork(nLayers, nNodes, activation=tf.nn.sigmoid, \
                             wInit='normal', bInit='normal')
    regress.train(nEpochs)
    
    
def getStillingerWeber():

    # parameters                            
    A = 7.049556277
    B = 0.6022245584
    p = 4.0
    q = 0.0
    a = 1.80
    Lambda = 21.0
    gamma = 1.20
    cosC = -1.0/3
    epsilon = 2.1683
    sigma = 2.0951  
    
    # Stillinger-Weber            
    function = lambda Rij, Rik, cosTheta:  epsilon*A*(B*(sigma/Rij)**p - (sigma/Rij)**q) * \
                                           np.exp(sigma / (Rij - a*sigma)) + \
                                           np.sum( epsilon*Lambda*(cosTheta - cosC)**2 * \
                                                   np.exp( (gamma*sigma) / (Rij - a*sigma) ) * \
                                                   np.exp( (gamma*sigma) / (Rik - a*sigma) ) )
                                                   
    return function, a, sigma

    
    
def StillingerWeberSymmetry(trainSize, batchSize, testSize, nLayers, nNodes, nEpochs, \
                            neighbours, symmFunctype, method, 
                            outputs=1, varyingNeigh=True, filename=''):
    """
    Train neural network to simulate tetrahedral Si atoms
    methodGenerate random input training data or use xyz-data from lammps
    Output training data is calculated with my sw-potential, i.e.
    energies not from lammps
    method=threeBodySymmetry: random configs
    method=lammps: configs from lammps, but not energies
    """
    
    function, a, sigma = getStillingerWeber()
    
    # set limits for input training data
    low = 2.0           # checked with lammps Si simulation
    
    # cutoff = sigma*a according to http://lammps.sandia.gov/doc/pair_sw.html
    # subtract a small number to prevent division by zero
    tol = 1e-14
    high = sigma*a + sigma/np.log(tol)
    print high
    exit(1)

    inputs = 0
    
    # train                           
    regress = regression.Regression(function, trainSize, batchSize, testSize, inputs, outputs)
    regress.generateData(low, high, method, neighbours=neighbours, 
                         symmFuncType='G4', filename=filename, varyingNeigh=varyingNeigh)
    regress.constructNetwork(nLayers, nNodes, activation=tf.nn.sigmoid, \
                             wInit='normal', bInit='normal')
    regress.train(nEpochs)
    
    
def lammpsTrainingSi(nLayers, nNodes, nEpochs, symmFuncType, filename, outputs=1, activation=tf.nn.sigmoid, 
                     useFunction=False, forces=False):
    """
    Use neighbour data and energies from lammps with sw-potential 
    as input and output training data respectively
    """
               
    # get energies from sw lammps  
    if useFunction: 
        function, _, _ = getStillingerWeber()
    else:
        function = None    
  
    # these are sampled from lammps
    trainSize = batchSize = testSize = inputs = low = high = 0
                       
    regress = regression.Regression(function, trainSize, batchSize, testSize, inputs, outputs)
    regress.generateData(low, high, 'lammps', symmFuncType='G4', filename=filename, forces=forces)
    regress.constructNetwork(nLayers, nNodes, activation=activation, \
                             wInit='xavier', bInit='zeros')
    regress.train(nEpochs)
    
    
def lammpsTrainsSiO2(nLayers, nNodes, nEpochs, filename, outputs=1, activation=tf.nn.sigmoid, \
                     useFunction=False):
    """
    Use neighbour data and energies from lammps with vashista?-potential
    as input and output training data respectively
    """
    
    # these are sampled from lammps
    function = None
    trainSize = batchSize = testSize = inputs = low = high = 0  
    
    regress = regression.Regression(function, trainSize, batchSize, testSize, inputs, outputs)
    regress.generateData(low, high, 'lammps', filename=filename)
    regress.constructNetwork(nLayers, nNodes, activation=activation, \
                             wInit='xavier', bInit='zeros')
    regress.train(nEpochs)
    
    
    
    
    
    
               
    
#testActivations(int(1e6), int(1e4), int(1e3), 3, 5, 100000)
#LennardJonesNeighboursForce(int(1e5), int(1e4), int(1e3), 2, 100, int(2e6), 5)





""" trainSize, batchSize, testSize, nLayers, nNodes, nEpochs, """

"""LJ med en input og en output"""
#LennardJonesExample(int(1e6), int(1e4), int(1e3), 2, 4, 300000)

"""Lj med flere naboer"""
#LennardJonesNeighbours(int(1e5), int(1e4), int(1e3), 2, 40, int(1e5), 10)



"""trainSize, batchSize, testSize, nLayers, nNodes, nEpochs, nNeighbours, nSymmfuncs, symmFuncType (G1 or G2)"""

"""LJ med radielle symmetrifunksjoner"""
#LennardJonesSymmetryFunctions(int(5e3), int(1e3), int(5e2), 2, 70, int(1e6), 70, 70, 'G2', 
#                              varyingNeigh=False)

"""Stillinger Weber med angular symmetrifunksjoner og lammpsffff-data"""
#StillingerWeberSymmetry(int(1e3), int(3e2), int(1e2), 2, 35, int(1e6), 15, 'G4', 'threeBodySymmetry', \
#                        varyingNeigh=False)#, \
#                        filename="../LAMMPS_test/Silicon/Data/24.02-16.11.12/neighbours.txt")

"""Lammps Stillinger-Weber kjoeringer gir naboer og energier"""
lammpsTrainingSi(2, 35, int(2e6), 'G4', \
                 "../LAMMPS_test/Silicon/Data/24.02-16.11.12/neighbours.txt", \
                 activation=tf.nn.sigmoid, useFunction=True)
                        
                        
                        
                        
                        
                        
