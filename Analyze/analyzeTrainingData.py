""""
1. Visualize symmetry functions for different systems that are saved to file with symmetryParameters.py
2. Visualize and analyze symmetrized input data for a specific training session
"""

import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

import numpy as np
import matplotlib.pyplot as plt
import DataGeneration.readers as readers
import sys
import Tools.matplotlibParameters as pltParams

def cutoffFunction(R, Rc):   
    
    value = 0.5 * (np.cos(np.pi*R / Rc) + 1)

    # set elements above cutoff to zero so they dont contribute to sum
    if isinstance(R, np.ndarray):
        value[np.where(R > Rc)[0]] = 0
    else:
        if R > Rc:
            value = 0
        
    return value  
 
    
def G1(Rij, cutoff):
    
    return cutoffFunction(Rij, cutoff)
    
    
def G2(Rij, width, cutoff, center):
    
    return np.exp(-width*(Rij - center)**2) * cutoffFunction(Rij, cutoff)
    
    
def G3(Rij, cutoff, kappa):
    
    return np.cos(kappa*Rij) * cutoffFunction(Rij, cutoff)
    
    
def G4(Rij, Rik, Rjk, theta, width, cutoff, zeta, inversion):
    
    return 2**(1-zeta) * (1 + inversion*np.cos(theta))**zeta * \
           np.exp( -width*(Rij**2 + Rik**2 + Rjk**2) ) * \
           cutoffFunction(Rij, cutoff) * cutoffFunction(Rik, cutoff) * cutoffFunction(Rjk, cutoff)
          
          
def G5(Rij, Rik, cosTheta, width, cutoff, thetaRange, inversion):
    
    return 2**(1-thetaRange) * (1 + inversion*cosTheta)**thetaRange * \
           np.exp( -width*(Rij**2 + Rik**2) ) * \
           cutoffFunction(Rij, cutoff) * cutoffFunction(Rik, cutoff)
   
        
def G4G5angular(theta, zeta, inversion):
    
    return 2**(1-zeta) * (1 + inversion*np.cos(theta))**zeta
           
           


# set plot parameters


# flags
plotFlag = False
analyzeFlag = False
saveFlag = False

# parse arguments
saveFlag = False
saveFileName = ''
if len(sys.argv) > 1:
    i = 1
    while i < len(sys.argv):
        
        if sys.argv[i] == '--plot':
            i += 1
            plotFlag = True
            parametersName = sys.argv[i]
            i += 1
            
        elif sys.argv[i] == '--analyze':
            i += 1
            analyzeFlag = True
            trainingDir = sys.argv[i]
            i += 1
                        
        elif sys.argv[i] == '--save':
            i += 1
            saveFlag     = True
            saveFileName = sys.argv[i]
            i += 1
            
        else:
            i += 1

        
def plotSymmetryFunctions(parametersName, plotG2=False, plotG4=False, plotG5=False, plotAngular=False):

    # read parameters
    parameters = readers.readParameters(parametersName)
    
    # G2: eta - Rc - Rs
    # G4: eta - Rc - zeta - lambda
    
    # split parameters list into G2 and G4/G5
    parameters2 = []
    parameters3 = []
    
    for param in parameters:
        if len(param) == 3:
            parameters2.append(param)
        else:
            parameters3.append(param)
        
    globalCutoff = max(parameters[:][1])
    print "Global cutoff:", globalCutoff
    
    Rij2 = np.linspace(0, globalCutoff + 2, 1000)   
    
    pltParams.defineColormap(len(parameters2), plt.cm.jet)
    
    ##### G2 plot #####
    
    if plotG2:
        legends = []
        for eta, Rc, Rs in parameters2:
            functionValue = G2(Rij2, eta, Rc, Rs)
            functionValue[np.where(Rij2 > Rc)[0]] = 0
            plt.plot(Rij2, functionValue)
            # with units:
            #legends.append(r'$\eta=%1.3f \, \mathrm{\AA{}}^{-2}, R_c=%1.1f  \, \mathrm{\AA{}}, R_s=%1.1f \, \mathrm{\AA{}}$' % \
            #               (eta, Rc, Rs) )
            # without units and Rc:
            legends.append(r'$\eta=%1.3f, R_s=%1.1f$' % \
                           (eta, Rs) )              
            plt.hold('on')
        
        plt.legend(legends, prop={'size':20})
        plt.xlabel(r'$R_{ij}$')
        plt.ylabel(r'$G^2_i$')
        plt.tight_layout()
        if saveFlag:
            plt.savefig(saveFileName)
        else: 
            plt.show()
    
    
    ##### G4/G5 plot #####
    
    pltParams.defineColormap(len(parameters3), plt.cm.jet)
    
    plt.figure()
    
    # plot vs Rij
    Rij3 = np.linspace(0, globalCutoff + 2, 1000)
    Rik = 2.0
    theta = np.pi/4
    cosTheta = np.cos(theta)
    Rjk = np.sqrt(Rij3**2 + Rik**2 - 2*Rij3*Rik*cosTheta) 

    if plotG4:
        legends = []
        for eta, Rc, zeta, Lambda in parameters3:
            functionValue = G4(Rij3, Rik, Rjk, cosTheta, eta, Rc, zeta, Lambda)
            plt.plot(Rij3, functionValue)
            legends.append(r'$\eta=%1.3f \, \mathrm{\AA{}}^{-2}, R_c=%1.1f  \, \mathrm{\AA{}}, \zeta=%d \, \lambda=%d$' % 
                           (eta, Rc, zeta, Lambda) )
            plt.hold('on')
            
        plt.legend(legends, prop={'size':20})
        plt.xlabel(r'$R_{ij}$')
        plt.ylabel(r'$G_i^4$')
        if saveFlag:
            plt.savefig(saveFileName)
        else: 
            plt.show()
            plt.figure()
            

    if plotG5:
        legends = []
        for eta, Rc, zeta, Lambda in parameters3:
            functionValue = G5(Rij3, Rik, cosTheta, eta, Rc, zeta, Lambda)
            plt.plot(Rij3, functionValue)
            legends.append(r'$\eta=%1.3f \, \mathrm{\AA{}}^{-2}, R_c=%1.1f  \, \mathrm{\AA{}}, \zeta=%d \, \lambda=%d$' % 
                           (eta, Rc, zeta, Lambda) )
            plt.hold('on')
            
        plt.legend(legends, prop={'size':20})
        plt.xlabel(r'$R_{ij}$')
        plt.ylabel(r'$G_i^5$')
        #if saveFlag:
        #    plt.savefig(saveFileName)
        #else: 
        #    plt.show()
        #    plt.figure()
            
    
    # plot vs theta
    Rij3 = 2.0
    Rik = 2.0
    theta = np.linspace(0, 2*np.pi, 1000)
    thetaAngle = theta*180/np.pi
    cosTheta = np.cos(theta)
    Rjk = np.sqrt(Rij3**2 + Rik**2 - 2*Rij3*Rik*cosTheta) 
    
    if plotG4:
        legends = []
        for eta, Rc, zeta, Lambda in parameters3:
            functionValue = G4(Rij3, Rik, Rjk, cosTheta, eta, Rc, zeta, Lambda)
            plt.plot(thetaAngle, functionValue)
            #legends.append(r'$\eta=%1.3f \, \mathrm{\AA{}}^{-2}, R_c=%1.1f  \, \mathrm{\AA{}}, \zeta=%d \, \lambda=%d$' % 
            #               (eta, Rc, zeta, Lambda) )
            # without eta:
            legends.append(r'$R_c=%1.1f  \, \mathrm{\AA{}}, \zeta=%d \, \lambda=%d$' % 
                           (eta, Rc, zeta, Lambda) )
            plt.hold('on')
            
        plt.legend(legends, prop={'size':20})
        plt.xlabel(r'$\theta$')
        plt.ylabel(r'$G_i^4$')
        if saveFlag:
            plt.savefig(saveFileName)
        else: 
            plt.show()
            plt.figure()
            

    if plotG5:
        plt.figure()
        legends = []
        for eta, Rc, zeta, Lambda in parameters3:
            functionValue = G5(Rij3, Rik, cosTheta, eta, Rc, zeta, Lambda)
            plt.plot(thetaAngle, functionValue)
            #legends.append(r'$\eta=%1.3f \, \mathrm{\AA{}}^{-2}, R_c=%1.1f  \, \mathrm{\AA{}}, \zeta=%d \, \lambda=%d$' % 
            #               (eta, Rc, zeta, Lambda) )
            # without eta and lambda:
            legends.append(r'$R_c=%1.1f  \, \mathrm{\AA{}}, \zeta=%d$' % 
                           (Rc, zeta) )
            plt.hold('on')
            
        plt.legend(legends, prop={'size':18}, loc=9)
        plt.xlabel(r'$\theta_{jik}$')
        plt.ylabel(r'$G_i^5$')
        plt.axis([0, 360, 0, 1.3])
        if saveFlag:
            plt.savefig(saveFileName)
        else: 
            plt.show()
            plt.figure()
    
      
    ##### angular part of G4/G5 ##### 
    theta = np.linspace(0, 2*np.pi, 1000) 
    thetaAngle = theta*180/np.pi
    
    if plotAngular:
        legends = []
        for _, _, zeta, Lambda in parameters3:
            functionValue = G4G5angular(theta, zeta, Lambda)
            plt.plot(thetaAngle, functionValue)
            legends.append(r'$\zeta = %d, \, \lambda = %d$' % (zeta, Lambda))
            plt.hold('on')
            
        plt.legend(legends, prop={'size':20}, loc=9)
        plt.xlabel(r'$\theta$')
        plt.ylabel(r'$G^4/G^5$ angular part')
        plt.axis([0, 2*180, 0, 2])
        plt.tight_layout()
        if saveFlag:
            plt.savefig(saveFileName)
        else:
            plt.show()
        
def analyzeInputData(trainingDir, multiType=True, plotCoordsDist=True, plotSymmDist=True, atomType=0,
                     rangeLimit=0.1):
    
        # read meta file for given training session (to load training data)
        nNodes, nLayers, activation, inputs, outputs, lammpsDir = readers.readMetaFile(trainingDir + '/meta.dat')
        lammpsDir = '../' + lammpsDir
        
        # read symmetrized input data for given training session
        symmetryFileName = lammpsDir + 'symmetryBehlerklargerj.txt'
        print 'Symmetry filename:', symmetryFileName
        if os.path.isfile(symmetryFileName):
                print "Reading symmetrized data"
                inputData = readers.readSymmetryData(symmetryFileName)
        else: 
            print "Symmetry values file does not exist, has to be made"
            exit(1)
        
        if multiType:
            filename = "neighbours%d.txt" % atomType
            neighbourFile = lammpsDir + filename
            print "Reading multi-type training data"
            x, y, z, r, types, E = readers.readNeighbourDataMultiType(neighbourFile)
        else:
            filename = "neighbours.txt"
            neighbourFile = lammpsDir + filename
            print "Reading single-type training data"
            x, y, z, r, E = readers.readNeighbourData(neighbourFile)
          
          
        if plotCoordsDist:
            # gather all r in sample
            allX = []; allY = []; allZ = []
            allR = []
            for i in xrange(len(r)):
                for j in xrange(len(r[i])):
                    allX.append(x[i][j])
                    allY.append(y[i][j])
                    allZ.append(z[i][j])
                    allR.append(r[i][j])
        
            allR = np.sqrt(np.array(allR))
            plt.hist(allR, bins=100)
            plt.legend('r')
            plt.show()
            plt.figure()
            plt.hist(allX, bins=100)
            plt.legend('x')
            plt.show()
            plt.figure()
            plt.hist(allY, bins=100) 
            plt.legend('y')
            plt.show()
            plt.figure()
            plt.hist(allZ, bins=100)
            plt.legend('z')
            plt.show()
            plt.figure()
        
        if plotSymmDist:
            inputDataFlat = inputData.flatten()
            N = len(inputDataFlat)
            
            print "Average symmetry value: ", np.average(inputDataFlat)
            print "Max symmetry value: ", np.max(inputDataFlat)
            print "Min symmetry value: ", np.min(inputDataFlat)
            print "Fraction of zeros: ", len(np.where(inputDataFlat == 0)[0]) / float(N)
            
            #plt.hist(inputDataFlat, bins=100)
            #plt.show()
            
            plt.figure()
            
            plt.hist(inputData[:,30], bins=20)
            plt.show()
            
            # check for zeros
            numberOfSymmFuncs = len(inputData[0])
            numberOfTrainExamples = len(inputData)
            smallRange = []
            for i in xrange(numberOfSymmFuncs):
                symmFunc = inputData[:,i]
                
                print
                print 'Symmetry function', i
                print 'Fraction of zeros:', 1 - (np.count_nonzero(symmFunc) / float(numberOfTrainExamples))
                imax = np.max(symmFunc)
                imin = np.min(symmFunc)
                irange = imax - imin
                iave = np.mean(symmFunc)
                print 'Max:', imax
                print 'Min:', imin
                print 'Range:', irange
                print 'Average:', iave
                
                if irange < rangeLimit:
                    smallRange.append([i,irange])

                
            print smallRange
            print np.corrcoef(inputData[:,46], inputData[:,47])
        
##### main #####
        
        
if plotFlag: 
    plotSymmetryFunctions(parametersName, 
                          plotG2=True, 
                          plotG4=False,
                          plotG5=False,
                          plotAngular=False)
    
if analyzeFlag:
    analyzeInputData(trainingDir,
                     multiType=False, 
                     plotCoordsDist=False, 
                     plotSymmDist=True)
        
    





