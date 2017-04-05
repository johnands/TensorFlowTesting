import numpy as np
import tensorflow as tf
import DataGeneration.symmetries as symmetry
import DataGeneration.lammpsData as data

def func(x):

    return np.sum(x**2)
    
def cutoffFunctionNew(R, Rc, cut=False):   
    
    value = 0.5 * (tf.cos(np.pi*R / Rc) + 1)

    # set elements above cutoff to zero so they dont contribute to sum
    """if cut:
        if isinstance(R, np.ndarray):
            value[np.where(R > Rc)[0]] = 0
        else:
            if R > Rc:
                value = 0"""
        
    return value
    
def G4new(xij, yij, zij, xik, yik, zik, eta, Rc, zeta, Lambda):
       
    Rij2 = xij*xij + yij*yij + zij*zij
    Rij = tf.sqrt(Rij2)
    
    Rik2 = xik*xik + yik*yik + zik*zik
    Rik = tf.sqrt(Rik2)
    
    cosTheta = (xij*xik + yij*yik + zij*zik) / (Rij*Rik)

    xjk = xij - xik
    yjk = yij - yik
    zjk = zij - zik
    Rjk = tf.sqrt(xjk*xjk + yjk*yjk + zjk*zjk)

    
    return 2**(1-zeta) * tf.reduce_sum( (1 + Lambda*cosTheta)**zeta * \
           tf.exp( -eta*(Rij2 + Rik2 + Rjk*Rjk) ) * \
           cutoffFunctionNew(Rij, Rc) * cutoffFunctionNew(Rik, Rc) * cutoffFunctionNew(Rjk, Rc, cut=True) )
           
        
def G5new(xij, yij, zij, xik, yik, zik, eta, Rc, zeta, Lambda):
       
    Rij2 = xij*xij + yij*yij + zij*zij
    Rij = tf.sqrt(Rij2)
    
    Rik2 = xik*xik + yik*yik + zik*zik
    Rik = tf.sqrt(Rik2)
    
    cosTheta = (xij*xik + yij*yik + zij*zik) / (Rij*Rik)
   
    return 2**(1-zeta) * tf.reduce_sum( (1 + Lambda*cosTheta)**zeta * \
           tf.exp( -eta*(Rij2 + Rik2) ) * \
           cutoffFunctionNew(Rij, Rc) * cutoffFunctionNew(Rik, Rc) )
           
           
def diffG(symmFunc):
    
    with tf.Session() as sess:
    
        xij = tf.placeholder('float', [None, 1])
        yij = tf.placeholder('float', [None, 1])
        zij = tf.placeholder('float', [None, 1])
        
        # read neighbour data of 3-body system
        filename = "../LAMMPS_test/Silicon/Data/04.04-22.54.07/neighbours.txt"
        x, y, z, r, E = data.readNeighbourData(filename)
        
        # pick a neighbour list
        x = np.array(x[0])
        y = np.array(y[0])
        z = np.array(z[0])
        r = np.array(r[0])
              
        print "x: ", x
        print "y: ", y
        print "z: ", z
        print "r: ", r
        
        # pick j as first element of list, k is the rest
        k = np.arange(len(r[:])) > 0
        
        # number of True elements in k is the number of triplets
        nTriplets = np.sum(k)
        
        xik = tf.placeholder('float', [None, nTriplets])
        yik = tf.placeholder('float', [None, nTriplets])
        zik = tf.placeholder('float', [None, nTriplets])
            
        eta = 0.01
        Rc = 6
        zeta = 1
        Lambda = 1
        
        # xij: shape [1,1]
        # xik: shape [1,nTriplets]
        feed_dict = {xij: [[x[0]]], yij: [[y[0]]], zij: [[z[0]]], xik: [x[k]], yik: [y[k]], zik: [z[k]]}
        
        if symmFunc == 'G4':
            y = G4new(xij, yij, zij, xik, yik, zik, eta, Rc, zeta, Lambda)
        elif symmFunc == 'G5':
            y = G5new(xij, yij, zij, xik, yik, zik, eta, Rc, zeta, Lambda)
        
        gradient = tf.gradients(y, [xij, yij, zij, xik, yik, zik])
        
        tf.global_variables_initializer()
        
        print sess.run(y, feed_dict=feed_dict)    
        print sess.run(gradient, feed_dict=feed_dict)
    

    
diffG('G5')




    



