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

        
def plotSymmetryFunctions(parametersName, plotG2=False, plotG4=False, plotG5=False, plotAngular=False, 
                          radialDistName='', angularDistName=''):

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
    
    Rij2 = np.linspace(0, globalCutoff, 1000)   
    
    pltParams.defineColormap(len(parameters2), plt.cm.jet)
    
    # read radial dist
    if radialDistName:
        with open(radialDistName, 'r') as infile:
            binCentersRadial = []
            radialDist = []
            for line in infile:
                words = line.split()
                binCentersRadial.append(float(words[0]))
                radialDist.append(float(words[1]))
        
        binCentersRadial = np.array(binCentersRadial)        
        radialDist = np.array(radialDist)
        
    # read angular dist
    if angularDistName:
        with open(angularDistName, 'r') as infile:
            binCentersAngular = []
            angularDist = []
            for line in infile:
                words = line.split()
                binCentersAngular.append(float(words[0]))
                angularDist.append(float(words[1]))
        
        binCentersAngular = np.array(binCentersAngular)        
        angularDist = np.array(angularDist)
    
        
    
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
        
        if radialDistName:
            plt.plot(binCentersRadial, radialDist/np.max(radialDist), 'k--', linewidth=2)
            #legends.append('Radial distribution')
        #plt.legend(legends, prop={'size':20})
        plt.xlabel(r'$R_{ij} \; [\mathrm{\AA{}}]$')
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
        #fig = plt.figure()
        ax = plt.subplot(111)
        legends = []
        for _, _, zeta, Lambda in parameters3:
            functionValue = G4G5angular(theta, zeta, Lambda)
            ax.plot(thetaAngle, functionValue)
            legends.append(r'$\zeta = %d, \, \lambda = %d$' % (zeta, Lambda))
            #plt.hold('on')
            
        if angularDistName:
            ax.plot(binCentersAngular, angularDist, 'k--', linewidth=2)
            #legends.append('Angular distribution')
            

        # Shrink current axis by 20%
        #box = ax.get_position()
        #ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        
        # Put a legend to the right of the current axis
        #ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            
        #ax.legend(legends, prop={'size':18}, loc='center left', bbox_to_anchor=(1, 0.5))
        plt.xlabel(r'$\theta_{jik}$')
        plt.ylabel(r'$G^5$ angular part')
        plt.axis([0, 180, 0, 3.8])
        plt.tight_layout()
        if saveFlag:
            plt.savefig(saveFileName)
        else:
            plt.show()
        
def analyzeInputData(directory, multiType=False, plotRadialDist=False, plotSymmDist=False, atomType=0,
                     rangeLimit=0.1, plotSingleSymmDist=False, corrLimit=0.9, plotCorrelations=False, 
                     plotAngularDist=False, symmetryFile='/symmetryBehler.txt'):
    
        # read meta file for given training session (to load training data)
        metafile = directory + '/meta.dat'
        if os.path.isfile(metafile):      
            print 'Reading meta file', metafile
            nNodes, nLayers, activation, inputs, outputs, lammpsDir = readers.readMetaFile(metafile)
            lammpsDir = '../' + lammpsDir
            symmetryFileName = lammpsDir + symmetryFile
            
        # or read neighbour file if meta file does not exist, i.e. no NN has been trained on this data
        else:
            lammpsDir = directory
            symmetryFileName = directory + symmetryFile
            print 'Reading symmetry file', directory, 'directly'
            
        
        # read symmetrized input data for given training session    
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
          
          
        if plotRadialDist or plotAngularDist:
            # gather all r in sample
            allX = []; allY = []; allZ = []
            allR = []
            for i in xrange(len(r)):
                for j in xrange(len(r[i])):
                    allX.append(x[i][j])
                    allY.append(y[i][j])
                    allZ.append(z[i][j])
                    allR.append(r[i][j])
        
        if plotRadialDist:   
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
        
        if plotAngularDist:
            # compute angles
            angles = []
            for i in xrange(len(r)):
            
                # convert to arrays
                xi = np.array(x[i][:])
                yi = np.array(y[i][:])
                zi = np.array(z[i][:])
                ri = np.sqrt(np.array(r[i][:]))
                
                nNeighbours = len(xi)
                angleStep = []         
                # loop over triplets
                for j in xrange(nNeighbours-1):
                
                    # atom j
                    rij = ri[j]
                    xij = xi[j]; yij = yi[j]; zij = zi[j]
                    
                    # all k > j
                    k = np.arange(len(ri[:])) > j 
                    rik = ri[k] 
                    xik = xi[k]; yik = yi[k]; zik = zi[k]
                
                    # compute cos(theta_ijk) and rjk
                    theta = np.arccos( (xij*xik + yij*yik + zij*zik) / (rij*rik) )
                    theta *= 180/np.pi
                
                    # add to this list of angles for this step
                    angleStep.append( theta.tolist() )
                
                # flatten list    
                angleStep = [item for sublist in angleStep for item in sublist]
                
                # add to total nested list
                angles.append(angleStep)
            
            # make histogram
            nBins = 50
            step = 500
            binEdges = np.linspace(0, 180, nBins+1)
            dist, _ = np.histogram(angles[step], bins=binEdges)
            binCenters = (binEdges[1:] + binEdges[:-1]) / 2.0
            plt.plot(binCenters, dist)
            plt.legend(['Angular distribution step %d' % step])
            plt.show()
            
            ##### total time-averaged histogram #####
            cumulativeAngles = np.zeros(nBins)
            for i in xrange(len(r)):
                for j in xrange(len(angles[i])):
                    for k in xrange(nBins):
                        if angles[i][j] < binEdges[k]:
                            cumulativeAngles[k] += 1
                            break

            # normalize
            cumulativeAngles /= len(r)
            plt.plot(binCenters, cumulativeAngles, 'g-')
            plt.legend(['Averaged angular distribution of data set'])
            plt.show()
            
            """with open('../../LAMMPS_test/Silicon/Dist/angularDist.txt', 'w') as outfile:
                for i in xrange(len(binCenters)):
                    outfile.write('%g %g' % (binCenters[i], cumulativeAngles[i]))
                    outfile.write('\n')"""
            
            
        
        # plot complete distribution of all symmetry functions
        if plotSymmDist:
            inputDataFlat = inputData.flatten()
            N = len(inputDataFlat)
            
            print "Average symmetry value: ", np.average(inputDataFlat)
            print "Max symmetry value: ", np.max(inputDataFlat)
            print "Min symmetry value: ", np.min(inputDataFlat)
            print "Fraction of zeros: ", len(np.where(inputDataFlat == 0)[0]) / float(N)
            
            nBins = 50
            plt.figure()
            plt.hist(inputDataFlat, bins=nBins)
            plt.legend(['Histogram of all symmetry functions, %d bins' % nBins])
            plt.show()
            
        
        # plote distribution of a single symmetry function
        numberOfSymmFuncs = len(inputData[0])
        nBins = 30
        #plt.ion() 
        if plotSingleSymmDist:
            for s in xrange(numberOfSymmFuncs):
                plt.figure()
                plt.hist(inputData[:,s], bins=nBins)
                plt.legend(['Histogram of symm func %d, %d bins' % (s, nBins)])
                plt.show()
                #raw_input('Press Return key to continue: ')
                print 'Std. dev.:', np.std(inputData[:,s])
            
            
        # find range of each symmetry function
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
            istd = np.std(symmFunc)
            print 'Max:', imax
            print 'Min:', imin
            print 'Range:', irange
            print 'Average:', iave
            print 'Std. dev.', istd
            
            # identify functions with small range
            if irange < rangeLimit:
                smallRange.append([i,irange])

        print
        print 'Symm funcs with a range smaller than', rangeLimit, ':'
        print smallRange
            
        
        # find correlation coefficient matrix and identify functions which have 
        # a strong correlation
        corrMatrix = np.corrcoef(np.transpose(inputData))
        
        correlations = []
        
        if (corrMatrix == np.transpose(corrMatrix)).all:
            print 'Symmetric correlation coefficient matrix'
        
        for i in xrange(numberOfTrainExamples):
            for j in xrange(i+1,numberOfSymmFuncs):
                correlations.append(corrMatrix[i,j])
                if corrMatrix[i,j] > corrLimit:
                    print '(%d,%d): %1.3f' % (i, j, corrMatrix[i,j])
                    pass
               
        # histogram of correlations
        if plotCorrelations:
            plt.figure()
            nBins = 30
            plt.hist(correlations, bins=nBins)
            plt.legend(['Histogram of correlations between symm funcs, %d bins' % nBins])
            plt.show()
        
##### main #####
        
        
if plotFlag: 
    plotSymmetryFunctions(parametersName, 
                          plotG2=False, 
                          plotG4=False,
                          plotG5=True,
                          plotAngular=True, 
                          radialDistName='../../LAMMPS_test/Silicon/Dist/radialDist.txt', 
                          angularDistName='../../LAMMPS_test/Silicon/Dist/angularDist.txt')
    
if analyzeFlag:
    analyzeInputData(trainingDir,
                     multiType=False, 
                     plotRadialDist=False,
                     plotAngularDist=False,
                     plotSymmDist=True,
                     plotSingleSymmDist=False,
                     plotCorrelations=False,
                     symmetryFile='symmetryCustomShifted.txt', 
                     corrLimit=0.995)
        
    





