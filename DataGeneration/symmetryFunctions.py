import numpy as np

def cutoffFunction(rVector, cutoff, cut=False):   
    
    value = 0.5 * (np.cos(np.pi*rVector / cutoff) + 1)

    # set elements above cutoff to zero so they dont contribute to sum
    if cut:
        value[np.where(rVector > cutoff)[0]] = 0
        
    return value
 
    
def G1(Rij, cutoff):
    
    return np.sum(cutoffFunction(Rij, cutoff))
    
    
def G2(Rij, cutoff, width, center):
    
    return np.sum( np.exp(-width*(Rij - center)**2) * cutoffFunction(Rij, cutoff) )
    
    
def G3(Rij, kappa, cutoff):
    
    return np.sum( np.cos(kappa*Rij) * cutoffFunction(Rij, cutoff))
    
    
def G4(Rij, Rik, Rjk, theta, thetaRange, width, cutoff, inversion):
    
    return 2**(1-thetaRange) * np.sum( (1 + inversion*np.cos(theta))**thetaRange * \
           np.exp( -width*(Rij**2 + Rik**2 + Rjk**2) ) * \
           cutoffFunction(Rij, cutoff) * cutoffFunction(Rik, cutoff) * cutoffFunction(Rjk, cutoff, cut=True) )
           
           
def G5(Rij, Rik, theta, thetaRange, width, cutoff, inversion):
    
    return 2**(1-thetaRange) * np.sum( (1 + inversion*np.cos(theta))**thetaRange * \
           np.exp( -width*(Rij**2 + Rik**2) ) * \
           cutoffFunction(Rij, cutoff) * cutoffFunction(Rik, cutoff) )