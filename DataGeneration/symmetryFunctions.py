import numpy as np

def cutoffFunction(rVector, cutoff, cut=False):   
    
    value = 0.5 * (np.cos(np.pi*rVector / cutoff) + 1)

    # set elements above cutoff to zero so they dont contribute to sum
    if cut:
        value[np.where(rVector > cutoff)[0]] = 0
        
    return value
 
    
def G1(rVector, cutoff):
    
    return np.sum(cutoffFunction(rVector, cutoff))
    
    
def G2(rVector, cutoff, width, center):
    
    return np.sum( np.exp(-width*(rVector - center)**2) * cutoffFunction(rVector, cutoff) )
    
    
def G3(Rij, Rik, Rjk, theta, thetaRange, width, cutoff, inversion):
    
    return 2**(1-thetaRange) * np.sum( (1 + inversion*np.cos(theta))**thetaRange * \
           np.exp(-width*(Rij**2 + Rik**2 + Rjk**2)) * \
           cutoffFunction(Rij, cutoff) * cutoffFunction(Rik, cutoff) * cutoffFunction(Rjk, cutoff, cut=True) )