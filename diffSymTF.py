import numpy as np
import tensorflow as tf
import DataGeneration.lammpsData as data
import DataGeneration.symmetryFunctions as symmFuncs

np.set_printoptions(precision=18)
           

def readCoordinates(filename):
    
    with open(filename, 'r') as infile:
        
        x = []; y = []; z = []
        for line in infile:
            words = line.split()
            N = len(words) / 3
            xi = []; yi = []; zi = [];
            for i in xrange(N):
                xi.append(float(words[3*i]))
                yi.append(float(words[3*i+1]))
                zi.append(float(words[3*i+2]))
            
            x.append(xi)
            y.append(yi)
            z.append(zi)
                
    return x, y, z
    
def readDerivatives(filename):
    
    with open(filename, 'r') as infile:
        
        dGj = []; dGk = []
        j = True
        for line in infile:
            words = line.split()
            
            if j:
                dGjline = []
                dGjline.append(float(words[0]))
                dGjline.append(float(words[1]))
                dGjline.append(float(words[2]))
                dGj.append(dGjline)
                j = False
            else:
                dGkline = []
                dGkline.append(float(words[0]))
                dGkline.append(float(words[1]))
                dGkline.append(float(words[2]))
                dGk.append(dGkline)
                j = True
                
    return np.array(dGj), np.array(dGk)
                     
    
    
def readParameters(filename):
    
    parameters = []
    with open(filename, 'r') as infile:
        infile.readline()
        for line in infile:
            param = []
            words = line.split()
            for word in words:
                param.append(float(word))
            parameters.append(param)
            
    return parameters
        
              
           
def diffG(symmFunc, sympy):
    
    with tf.Session() as sess:
    
        xij = tf.placeholder(tf.float64, [None, 1])
        yij = tf.placeholder(tf.float64, [None, 1])
        zij = tf.placeholder(tf.float64, [None, 1])
        
        """
        I have summed up dG5dj and dG5dk for certain steps in an MD simulation both for
        the sympy expression and my hard-coded expression. 
        The derivatives are only summed up for ONE rij, i.e. when l=0 in the triplet force loop
        """
        if sympy:
            filenameConfigs = "../LAMMPS_test/TestNN/Tests/Gderivative/sympyConfigs.txt"
            filenameDerivatives = "../LAMMPS_test/TestNN/Tests/Gderivative/sympyDerivatives.txt"
        else:
            filenameConfigs = "../LAMMPS_test/TestNN/Tests/Gderivative/myConfigs.txt"
            filenameDerivatives = "../LAMMPS_test/TestNN/Tests/Gderivative/myDerivatives.txt"
      
        x, y, z = readCoordinates(filenameConfigs)
        dGj, dGk = readDerivatives(filenameDerivatives)
                 
        nSamples = len(x)
        print "Number of samples: ", nSamples
            
        parameters = readParameters("TrainingData/21.04-19.05.10/parameters.dat")
        
        # loop through all configs and find sum of derivative for all symm funcs
        derivativesj = np.zeros((nSamples, 3))
        derivativesk = np.zeros((nSamples, 3))
        for i in xrange(nSamples):
            
            xi = np.array(x[i])
            yi = np.array(y[i])
            zi = np.array(z[i])
            print xi
            print yi
            print zi
            
            # pick j as first element of list, k is the rest
            k = np.arange(len(xi[:])) > 0
            
            # number of True elements in k is the number of triplets
            nTriplets = np.sum(k)
            print "Number of triplets: ", nTriplets
            
            if not nTriplets > 0:
                continue
            
            xik = tf.placeholder(tf.float64, [None, nTriplets])
            yik = tf.placeholder(tf.float64, [None, nTriplets])
            zik = tf.placeholder(tf.float64, [None, nTriplets])
                
            # xij: shape [1,1]
            # xik: shape [1,nTriplets]
            # pick the i-th neighbour list
            feed_dict = {xij: [[xi[0]]], yij: [[yi[0]]], zij: [[zi[0]]], xik: [xi[k]], yik: [yi[k]], zik: [zi[k]]}
            
            for j, param in enumerate(parameters):
                
                if len(param) != 4:
                    continue
                
                if symmFunc == 'G4':
                    func = symmFuncs.G4TF(xij, yij, zij, xik, yik, zik, param[0], param[1], param[2], param[3])
                elif symmFunc == 'G5':
                    func = symmFuncs.G5TF(xij, yij, zij, xik, yik, zik, param[0], param[1], param[2], param[3])
        
                gradient = tf.gradients(func, [xij, yij, zij, xik, yik, zik])
              
                yes  = sess.run(gradient, feed_dict=feed_dict)
                derivativesj[i,0] += yes[0][0]
                derivativesj[i,1] += yes[1][0]
                derivativesj[i,2] += yes[2][0]
                
                dkx = np.array(yes[3])
                dky = np.array(yes[4])
                dkz = np.array(yes[5])
                
                derivativesk[i,0] += np.sum(dkx)
                derivativesk[i,1] += np.sum(dky)
                derivativesk[i,2] += np.sum(dkz)
    
    diffxj = derivativesj[:][0] - dGj[:][0]
    diffyj = derivativesj[:][1] - dGj[:][1]
    diffzj = derivativesj[:][2] - dGj[:][2]
    
    diffxk = derivativesk[:][0] - dGk[:][0]
    diffyk = derivativesk[:][1] - dGk[:][1]
    diffzk = derivativesk[:][2] - dGk[:][2]
    
    print "Sum of squares dG/dxj: ", np.sum(diffxj**2)
    print "Sum of squares dG/dyj: ", np.sum(diffyj**2)
    print "Sum of squares dG/dzj: ", np.sum(diffzj**2)
    print "Sum of squares dG/dxk: ", np.sum(diffxk**2)
    print "Sum of squares dG/dyk: ", np.sum(diffyk**2)
    print "Sum of squares dG/dzk: ", np.sum(diffzk**2)
    
    MADxj = np.mean(np.abs(diffxj))
    MADyj = np.mean(np.abs(diffyj))
    MADzj = np.mean(np.abs(diffzj))
    MADxk = np.mean(np.abs(diffxk))
    MADyk = np.mean(np.abs(diffyk))
    MADzk = np.mean(np.abs(diffzk))

    print "Mean absolute deviation dG/dxj: ", MADxj
    print "Mean absolute deviation dG/dyj: ", MADyj
    print "Mean absolute deviation dG/dzj: ", MADzj
    print "Mean absolute deviation dG/dxk: ", MADxk
    print "Mean absolute deviation dG/dyk: ", MADyk
    print "Mean absolute deviation dG/dzk: ", MADzk
    
    MADTotal = (MADxj + MADyj + MADzj + MADxk + MADyk + MADzk) / 6.0
    print "Total mean absolute deviation: ", MADTotal
    
    

    
diffG('G5', True)




    



