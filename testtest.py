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
           tf.exp( -eta*(Rij**2 + Rik**2 + Rjk**2) ) * \
           cutoffFunctionNew(Rij, Rc) * cutoffFunctionNew(Rik, Rc) * cutoffFunctionNew(Rjk, Rc, cut=True) )
    
    
    

with tf.Session() as sess:

    inputs = 1
    xij = tf.placeholder('float', [None, 1])
    yij = tf.placeholder('float', [None, 1])
    zij = tf.placeholder('float', [None, 1])
    
    xik = tf.placeholder('float', [None, inputs])
    yik = tf.placeholder('float', [None, inputs])
    zik = tf.placeholder('float', [None, inputs])
    
    x0 = np.array([1.360657349, -1.382538701, -1.356904817, 1.347116757])
    y0 = np.array([1.377881253, -1.325733246, 1.378576362, -1.334633321])
    z0 = np.array([1.313130941, 1.293225662, -1.387754934, -1.412519148]) 
    r0 = x0**2 + y0**2 + z0**2
    
    #filename = "../LAMMPS_test/Silicon/Data/21.03-11.56.37/neighbours.txt"
    #x, y, z, r, E = data.readNeighbourData(filename)
    
    """x0 = np.array(x[4])
    y0 = np.array(y[4])
    z0 = np.array(z[4])
    r0 = np.array(r[4])"""
    print "x: ", x0
    print "y: ", y0
    print "z: ", z0
    print "r: ", r0
    

    
    eta = 0.01
    Rc = 6
    zeta = 1
    Lambda = 1
    
    #k = np.arange(len(r0[:])) > 0
    k = 1
    
    """xij = x0[0]
    yij = y0[0]
    zij = z0[0]
    xik = x0[k]
    yik = y0[k]
    zik = z0[k]
    
    print "xij: ", xij
    print "yij: ", yij
    print "zij: ", zij
    print "xik: ", xik
    print "yik: ", yik
    print "zik: ", zik
       
    Rij2 = xij*xij + yij*yij + zij*zij
    Rij = np.sqrt(Rij2)
    
    print "Rij: ", Rij
    
    Rik2 = xik*xik + yik*yik + zik*zik
    Rik = np.sqrt(Rik2)
    
    print "Rik: ", Rik
    
    cosTheta = (xij*xik + yij*yik + zij*zik) / (Rij*Rik)
    
    print "cosTheta: ", cosTheta
    
    xjk = xij - xik
    yjk = yij - yik
    zjk = zij - zik
    Rjk = np.sqrt(xjk*xjk + yjk*yjk + zjk*zjk)
    
    print "xjk: ", xjk
    print "yjk: ", yjk
    print "zjk: ", zjk
    print "Rjk: ", Rjk
    exit(1)"""
    
    # xij: shape [1,1]
    # xik: shape [1,inputs] or [1,len(k)]
    # now k is a number, if a list: remove the outermost brackets on xik, yik and zik
    feed_dict = {xij: [[x0[0]]], yij: [[y0[0]]], zij: [[z0[0]]], xik: [[x0[k]]], yik: [[y0[k]]], zik: [[z0[k]]]}
    
    y = G4new(xij, yij, zij, xik, yik, zik, eta, Rc, zeta, Lambda)
    
    gradient = tf.gradients(y, [xij, yij, zij, xik, yik, zik])
    
    tf.global_variables_initializer()
    
    print sess.run(y, feed_dict=feed_dict)    
    print sess.run(gradient, feed_dict=feed_dict)




    



