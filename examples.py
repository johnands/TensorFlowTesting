import regression
import numpy as np
import tensorflow as tf
import time

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
        

def setUpLJPotential(units, shifted=False, derivative=False):
    
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
        
    if derivative:
        functionDerivative = lambda s: -24*epsilon*(2*(sigma**12/s**13) - (sigma**6/s**7))
        
    return function, a, cutoff, functionDerivative
        

 
def LennardJonesExample(trainSize=int(1e5), batchSize=50, testSize=int(1e4), 
                        nLayers=1, nNodes=10, nEpochs=int(1e5), 
                        units='metal', activation=tf.nn.sigmoid):
    """
    Train to reproduce shifted L-J potential to 
    verify implementation of network and backpropagation in the MD code
    This is a 1-dimensional example
    """

    function, a, cutoff, functionDerivative = setUpLJPotential('metal', derivative=True)    
    
    regress = regression.Regression(function, trainSize, batchSize, testSize, 1, 1, 
                                    functionDerivative=functionDerivative)
    regress.generateData(a, cutoff, 'twoBody')
    regress.constructNetwork(nLayers, nNodes, activation=activation, \
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
    

def getTwoBodySW(): 
    
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
    function = lambda Rij:  epsilon*A*(B*(sigma/Rij)**p - (sigma/Rij)**q) * \
                            np.exp(sigma / (Rij - a*sigma))
                                                   
    return function, a, sigma

    
    
def StillingerWeberSymmetry(trainSize, batchSize, testSize, nLayers, nNodes, nEpochs, \
                            neighbours, symmFunctype, method, 
                            outputs=1, varyingNeigh=True, dataFolder=''):
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
                         symmFuncType='G4', dataFolder=dataFolder, varyingNeigh=varyingNeigh)
    regress.constructNetwork(nLayers, nNodes, activation=tf.nn.sigmoid, \
                             wInit='normal', bInit='normal')
    regress.train(nEpochs)
    

def lammpsTrainingSiO2(nLayers=2, nNodes=10, nEpochs=int(1e5), symmFuncType='G5', 
                       activation= tf.nn.sigmoid, lammpsDir='4Atoms/T1e3N1e4', forces=False, 
                       batch=5, learningRate=0.001, RMSEtol=0.003, outputs=1, atomType=0, nTypes=2, nAtoms=10):
    """
    Use neighbour data and energies from lammps with vashista-potential
    as input and output training data respectively
    """
    
    lammpsDir = "../LAMMPS_test/Quartz/Data/TrainingData/" + lammpsDir + '/'
    
    # these are sampled from lammps
    function = None
    trainSize = batchSize = testSize = inputs = low = high = 0  
    
    regress = regression.Regression(function, trainSize, batchSize, testSize, inputs, outputs,
                                    learningRate=learningRate, RMSEtol=RMSEtol)
    regress.generateData(low, high, 'lammpsSiO2', 
                         symmFuncType=symmFuncType, dataFolder=lammpsDir, forces=forces, batch=batch,
                         atomType=atomType, nTypes=nTypes, nAtoms=nAtoms)
    regress.constructNetwork(nLayers, nNodes, activation=activation,
                             wInit='xavier', bInit='constant')
    regress.train(nEpochs)
    
    
    
    
def lammpsTrainingSi(nLayers=2, nNodes=35, nEpochs=int(1e5), symmFuncType='G5', \
                     lammpsDir='', outputs=1, activation=tf.nn.sigmoid, \
                     useFunction=False, forces=False, batch=5, Behler=True, \
                     klargerj=False, tags=False, learningRate=0.001, RMSEtol=1e-10, nTypes=1, 
                     normalize=False, shiftMean=False, standardize=False, 
                     wInit='uniform', bInit='zeros', constantValue=0.1, stdDev=0.1):
    """
    Use neighbour data and energies from lammps with sw-potential 
    as input and output training data respectively
    """
    
    lammpsDir = "../LAMMPS_test/Silicon/Data/TrainingData/" + lammpsDir + '/'
               
    # get energies from sw lammps  
    if useFunction: 
        function, _, _ = getStillingerWeber()
        #function, _, _ = getTwoBodySW()
    else:
        function = None    
  
    # these are sampled from lammps
    trainSize = batchSize = testSize = inputs = low = high = 0
                       
    regress = regression.Regression(function, trainSize, batchSize, testSize, inputs, outputs,
                                    learningRate=learningRate, RMSEtol=RMSEtol)
    regress.generateData(low, high, 'lammpsSi', 
                         symmFuncType=symmFuncType, dataFolder=lammpsDir, forces=forces, batch=batch, 
                         Behler=Behler, klargerj=klargerj, tags=tags, nTypes=nTypes, 
                         normalize=normalize, shiftMean=shiftMean, standardize=standardize)
    regress.constructNetwork(nLayers, nNodes, activation=activation,
                             wInit=wInit, bInit=bInit, constantValue=constantValue, stdDev=stdDev)
    regress.train(nEpochs)
    
    
    
def gridSearchSi(maxLayers=3, minNodes=5, skipNodes=2, maxNodes=30, maxEpochs=1e5, symmFuncType='G5', \
                  lammpsDir='', outputs=1, activation=tf.nn.sigmoid, \
                  useFunction=False, forces=False, batch=5, Behler=True, \
                  klargerj=False, tags=False, learningRate=0.001, RMSEtol=1e-10, nTypes=1, 
                  normalize=False, shiftMean=False, standardize=False,
                  wInit='uniform', bInit='zeros', constantValue=0.1, stdDev=0.1):
    """
    Do a grid search to find a suitable NN architecture
    """
    
    lammpsDir = "../LAMMPS_test/Silicon/Data/TrainingData/" + lammpsDir + '/'  
    function = None
    
    # these are sampled from lammps
    trainSize = batchSize = testSize = inputs = low = high = 0
    regress = regression.Regression(function, trainSize, batchSize, testSize, inputs, outputs,
                                    learningRate=learningRate, RMSEtol=RMSEtol)
    regress.generateData(low, high, 'lammpsSi', 
                         symmFuncType=symmFuncType, dataFolder=lammpsDir, forces=forces, batch=batch, 
                         Behler=Behler, klargerj=klargerj, tags=tags, nTypes=nTypes, 
                         normalize=normalize, shiftMean=shiftMean, standardize=standardize)
                         
    # finding optimal value
    counter = 0
    for layers in xrange(1, maxLayers+1):
        for nodes in xrange(minNodes, maxNodes+1, skipNodes):
            regress.constructNetwork(layers, nodes, activation=activation, 
                                     wInit='uniform', bInit='zeros')
            testRMSE, epoch, timeElapsed = regress.train(maxEpochs)
            print "Layers: %2d, nodes: %2d, RMSE = %g, Epoch = %d, time = %10g" % (layers, nodes, testRMSE, epoch, timeElapsed)
            print
    
            if counter == 0:
                with open('Tests/timeElapsed.txt', 'w') as outFile:
                    outStr = "Timing analysis"
                    outFile.write(outStr + '\n')
                    
            with open('Tests/timeElapsed.txt', 'a') as outFile:
                outStr = "Layers: %2d, nodes: %2d, RMSE: %g, Epoch: %d, time = %10g" % (layers, nodes, testRMSE, epoch, timeElapsed)
                outFile.write(outStr + '\n')
            
            counter += 1
            
        
"""Lammps Stillinger-Weber gir naboer og energier"""
"""lammpsTrainingSi( nLayers       = 1, 
                  nNodes        = 10, 
                  nEpochs       = int(1e5), 
                  activation    = tf.nn.sigmoid, 
                  symmFuncType  = 'G5', 
                  lammpsDir     = 'Bulk/SiPotential/NNPotentialRun3',
                  Behler        = False, 
                  klargerj      = True, 
                  useFunction   = False, 
                  forces        = False, 
                  tags          = False,
                  batch         = 100, 
                  learningRate  = 0.005, 
                  RMSEtol       = 0.0005,
                  wInit         = 'uniform',
                  bInit         = 'zeros',
                  constantValue = 4.0,
                  normalize     = False, 
                  shiftMean     = True, 
                  standardize   = False )"""
    

"""Si grid searh"""
gridSearchSi(     maxLayers     = 2, 
                  minNodes      = 4,
                  skipNodes     = 4,
                  maxNodes      = 32, 
                  maxEpochs     = int(4e4),
                  RMSEtol       = 0.000001,  
                  activation    = tf.nn.sigmoid, 
                  symmFuncType  = 'G5', 
                  lammpsDir     = 'Bulk/SiPotential/NNPotentialRun3',
                  Behler        = False, 
                  klargerj      = True, 
                  useFunction   = False, 
                  forces        = False, 
                  tags          = False,
                  batch         = 200, 
                  learningRate  = 0.005, 
                  wInit         = 'uniform',
                  bInit         = 'zeros',
                  normalize     = False, 
                  shiftMean     = True, 
                  standardize   = False )
    
    
    
    

    
    

               
    
#testActivations(int(1e6), int(1e4), int(1e3), 3, 5, 100000)
#LennardJonesNeighboursForce(int(1e5), int(1e4), int(1e3), 2, 100, int(2e6), 5)





""" trainSize, batchSize, testSize, nLayers, nNodes, nEpochs, """

"""LJ med en input og en output"""
"""LennardJonesExample( trainSize = int(1e5), 
                     batchSize = int(200),
                     testSize  = int(1e3), 
                     nLayers   = 1, 
                     nNodes    = 10, 
                     nEpochs   = int(5e3) )"""


"""Lj med flere naboer"""
#LennardJonesNeighbours(int(1e5), int(1e4), int(1e3), 2, 40, int(1e5), 10)



"""trainSize, batchSize, testSize, nLayers, nNodes, nEpochs, nNeighbours, nSymmfuncs, symmFuncType"""

"""LJ med radielle symmetrifunksjoner"""
#LennardJonesSymmetryFunctions(int(5e3), int(1e3), int(5e2), 2, 70, int(1e6), 70, 70, 'G2', 
#                              varyingNeigh=False)

"""Stillinger Weber med angular symmetrifunksjoner og lammps-data"""
#StillingerWeberSymmetry(int(1e3), int(3e2), int(1e2), 2, 35, int(1e6), 15, 'G4', 'threeBodySymmetry', \
#                        varyingNeigh=False)#, \
#                        filename="../LAMMPS_test/Silicon/Data/24.02-16.11.12/neighbours.txt")

                  
"""Lammps Vashishta gir naboer og energier"""
"""lammpsTrainingSiO2( nLayers       = 2, 
                    nNodes        = 50, 
                    nEpochs       = int(1e5), 
                    activation    = tf.nn.sigmoid, 
                    symmFuncType  = 'G5', 
                    lammpsDir     = 'Bulk/L1T1000N1000NoAlgoNonPeriodic', 
                    atomType      = 1,
                    nTypes        = 2,
                    nAtoms        = 9,
                    forces        = False, 
                    batch         = 5, 
                    learningRate  = 0.001, 
                    RMSEtol       = 0.003 )"""
                    
# set numberOfEpochs = -1 when loading network to evalute it
# set nAtoms to 9 or more to signify training of bulk SiO2, i.e. use bulk symmetry parameters
                        
                        
                        
                        
                        
                        
