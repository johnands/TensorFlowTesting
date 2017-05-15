import numpy as np
import tensorflow as tf

def cutoffFunction(R, Rc):   
    
    value = 0.5 * (np.cos(np.pi*R / Rc) + 1)

    # set elements above cutoff to zero so they dont contribute to sum
    if isinstance(R, np.ndarray):
        value[np.where(R > Rc)[0]] = 0
    else:
        if R > Rc:
            value = 0
        
    return value   
 
    
def G1(Rij, Rc):
    
    return np.sum(cutoffFunction(Rij, Rc))
    
    
def G2(Rij, eta, Rc, Rs):
    
    return np.sum( np.exp(-eta*(Rij - Rs)**2) * cutoffFunction(Rij, Rc) )
    
    
def G3(Rij, Rc, kappa):
    
    return np.sum( np.cos(kappa*Rij) * cutoffFunction(Rij, Rc))
    
    
def G4(Rij, Rik, Rjk, cosTheta, eta, Rc, zeta, Lambda):
    
    return 2**(1-zeta) * np.sum( (1 + Lambda*cosTheta)**zeta * \
           np.exp( -eta*(Rij**2 + Rik**2 + Rjk**2) ) * \
           cutoffFunction(Rij, Rc) * cutoffFunction(Rik, Rc) * cutoffFunction(Rjk, Rc) )
           
           
def G5(Rij, Rik, cosTheta, eta, Rc, zeta, Lambda):
    
    return 2**(1-zeta) * np.sum( (1 + Lambda*cosTheta)**zeta* \
           np.exp( -eta*(Rij**2 + Rik**2) ) * \
           cutoffFunction(Rij, Rc) * cutoffFunction(Rik, Rc) )
           
           
           
def cutoffFunctionTF(R, Rc, cut=False):   
    
    value = 0.5 * (tf.cos(np.pi*R / Rc) + 1)

    # set elements above cutoff to zero so they dont contribute to sum
    """if cut:
        if isinstance(R, np.ndarray):
            value[np.where(R > Rc)[0]] = 0
        else:
            if R > Rc:
                value = 0"""
        
    return value
    
    
def G2TF(xij, yij, zij, eta, Rc, Rs):
    
    Rij = tf.sqrt(xij*xij + yij*yij + zij*zij)
    
    return tf.reduce_sum( tf.exp(-eta*(Rij - Rs)**2) * cutoffFunctionTF(Rij, Rc) )
    
    
def G4TF(xij, yij, zij, xik, yik, zik, eta, Rc, zeta, Lambda):
       
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
           cutoffFunctionTF(Rij, Rc) * cutoffFunctionTF(Rik, Rc) * cutoffFunctionTF(Rjk, Rc) )
           
        
def G5TF(xij, yij, zij, xik, yik, zik, eta, Rc, zeta, Lambda):
       
    Rij2 = xij*xij + yij*yij + zij*zij
    Rij = tf.sqrt(Rij2)
    
    Rik2 = xik*xik + yik*yik + zik*zik
    Rik = tf.sqrt(Rik2)
    
    cosTheta = (xij*xik + yij*yik + zij*zik) / (Rij*Rik)
   
    return 2**(1-zeta) * tf.reduce_sum( (1 + Lambda*cosTheta)**zeta * \
           tf.exp( -eta*(Rij2 + Rik2) ) * \
           cutoffFunctionTF(Rij, Rc) * cutoffFunctionTF(Rik, Rc) )
           
           
def dfcdr(R, Rc):
    
    return -0.5*(np.pi/Rc) * np.sin((np.pi*R) / Rc)
          
          
def dG2dr(xij, yij, zij, Rij, eta, Rc, Rs):
    
    dr = np.exp(-eta*(Rij - Rs)**2) * (2*eta*(Rs - Rij)*cutoffFunction(Rij, Rc) + dfcdr(Rij, Rc))
    fpair = -dr/Rij
    
    dij = []
    dij.append(fpair*xij)
    dij.append(fpair*yij)
    dij.append(fpair*zij)
    
    return dij
    
    
def dG5dj(xj, yj, zj, xk, yk, zk, Rj, Rk, CosTheta, \
          eta, Rc, zeta, Lambda):
    
    Rj2 = Rj*Rj
    Rk2 = Rk*Rk
    Fcj = cutoffFunction(Rj, Rc)
    Fck = cutoffFunction(Rk, Rc)
    dFcj = dfcdr(Rj, Rc)
    
    dGj = np.zeros((3,len(xk)))
    
    dGj[0] = 2**-zeta*Fck*(-2*Fcj*Lambda*zeta* \
        (CosTheta*Lambda + 1)**zeta*(CosTheta*Rk*xj - Rj*xk) - \
        4*Fcj*Rj**2*Rk*eta*xj*(CosTheta*Lambda + 1)**(zeta+1) + \
        2.0*Rj*Rk*dFcj*xj*(CosTheta*Lambda + 1)**(zeta+1))* \
        np.exp(-eta*(Rj2 + Rk2))/(Rj**2*Rk*(CosTheta*Lambda + 1))
    
    dGj[1] = 2**-zeta*Fck*(-2*Fcj*Lambda*zeta* \
        (CosTheta*Lambda + 1)**zeta*(CosTheta*Rk*yj - Rj*yk) - \
        4*Fcj*Rj**2*Rk*eta*yj*(CosTheta*Lambda + 1)**(zeta+1) + \
        2.0*Rj*Rk*dFcj*yj*(CosTheta*Lambda + 1)**(zeta+1))* \
        np.exp(-eta*(Rj2 + Rk2))/(Rj**2*Rk*(CosTheta*Lambda + 1))
    
    dGj[2] = 2**-zeta*Fck*(-2*Fcj*Lambda*zeta* \
        (CosTheta*Lambda + 1)**zeta*(CosTheta*Rk*zj - Rj*zk) - \
        4*Fcj*Rj**2*Rk*eta*zj*(CosTheta*Lambda + 1)**(zeta+1) + \
        2.0*Rj*Rk*dFcj*zj*(CosTheta*Lambda + 1)**(zeta+1))* \
        np.exp(-eta*(Rj2 + Rk2))/(Rj**2*Rk*(CosTheta*Lambda + 1))

    return -dGj


def dG5dk(xj, yj, zj, xk, yk, zk, Rj, Rk, CosTheta, \
          eta, Rc, zeta, Lambda):
    
    Rj2 = Rj*Rj
    Rk2 = Rk*Rk
    Fcj = cutoffFunction(Rj, Rc)
    Fck = cutoffFunction(Rk, Rc)
    dFck = dfcdr(Rk, Rc)
    
    dGk = np.zeros((3, len(xk)))
    
    dGk[0] = 2**-zeta*Fcj*(-2*Fck*Lambda*zeta* \
        (CosTheta*Lambda + 1)**zeta*(CosTheta*Rj*xk - Rk*xj) - \
        4*Fck*Rj*Rk2*eta*xk*(CosTheta*Lambda +1 )**(zeta+1) + \
        2.0*Rj*Rk*dFck*xk*(CosTheta*Lambda +1 )**(zeta+1))* \
        np.exp(-eta*(Rj2 + Rk2))/(Rj*Rk2*(CosTheta*Lambda + 1))
    
    dGk[1] = 2**-zeta*Fcj*(-2*Fck*Lambda*zeta* \
        (CosTheta*Lambda + 1)**zeta*(CosTheta*Rj*yk - Rk*yj) - \
        4*Fck*Rj*Rk2*eta*yk*(CosTheta*Lambda +1 )**(zeta+1) + \
        2.0*Rj*Rk*dFck*yk*(CosTheta*Lambda +1 )**(zeta+1))* \
        np.exp(-eta*(Rj2 + Rk2))/(Rj*Rk2*(CosTheta*Lambda + 1))
    
    dGk[2] = 2**-zeta*Fcj*(-2*Fck*Lambda*zeta* \
        (CosTheta*Lambda + 1)**zeta*(CosTheta*Rj*zk - Rk*zj) - \
        4*Fck*Rj*Rk2*eta*zk*(CosTheta*Lambda +1 )**(zeta+1) + \
        2.0*Rj*Rk*dFck*zk*(CosTheta*Lambda +1 )**(zeta+1))* \
        np.exp(-eta*(Rj2 + Rk2))/(Rj*Rk2*(CosTheta*Lambda + 1))
        
    return -dGk