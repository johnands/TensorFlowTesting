import numpy as np
import matplotlib.pyplot as plt

# define symmetry functions without the sum to plot

def cutoffFunction(rVector, cutoff, cut=False):   
    
    value = 0.5 * (np.cos(np.pi*rVector / cutoff) + 1)

    # set elements above cutoff to zero so they dont contribute to sum
    if cut:
        value[np.where(rVector > cutoff)[0]] = 0
        
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
           cutoffFunction(Rij, cutoff) * cutoffFunction(Rik, cutoff) * cutoffFunction(Rjk, cutoff, cut=True)
           
def G4angular(theta, zeta, inversion):
    
    return 2**(1-zeta) * (1 + inversion*np.cos(theta))**zeta
           
           
def G5(Rij, Rik, cosTheta, width, cutoff, thetaRange, inversion):
    
    return 2**(1-thetaRange) * (1 + inversion*cosTheta)**thetaRange * \
           np.exp( -width*(Rij**2 + Rik**2) ) * \
           cutoffFunction(Rij, cutoff) * cutoffFunction(Rik, cutoff)

##### test LJ #####

"""Rij = np.linspace(0, 12, 200)

widths = [0.05]#, 0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.1, 0.3, 0.7]
cutoffs = [8.5125]
centers = [0.0, 3.1, 4.5, 5.2, 5.9, 6.8, 7.8]

legends = []
for width in widths:
    for cutoff in cutoffs:
        for center in centers:   
            functionValue = G2(Rij, width, cutoff, center)
            functionValue[np.where(Rij > cutoff)[0]] = 0
            plt.plot(Rij, functionValue)
            legends.append(r'$\eta = %3.2f, R_c = %1.2f, R_s = %1.1f$' % (width, cutoff, center))
            plt.hold('on')
plt.legend(legends)    
plt.show()"""


##### test SW #####

# G2
"""Rij = np.linspace(0, 5, 200)

widthG2 = [0.01, 0.1, 1.0]
cutoffG2 = [5.0]
centerG2 = [0.0, 2.0, 4.0]

legends1 = []
for width in widthG2:
    for cutoff in cutoffG2:
        for center in centerG2:   
            functionValue = G2(Rij, width, cutoff, center)
            functionValue[np.where(Rij > cutoff)[0]] = 0
            plt.plot(Rij, functionValue)
            legends1.append(r'$\eta = %3.2f, R_c = %1.2f, R_s = %1.1f$' % (width, cutoff, center))
            plt.hold('on')
plt.legend(legends1)    
plt.show()"""

plt.figure()

plotTheta = True

# G4
if plotTheta:
    Rij = 3.0
    Rik = 2.0
    theta = np.linspace(0, 2*np.pi, 100)
    
else:
    Rij = np.linspace(0, 5, 200)
    Rik = 2.0 
    theta = np.pi/2

Rjk = np.sqrt(Rij**2 + Rik**2 - 2*Rij*Rik*np.cos(theta))     


widthG41 = [0.01]      
cutoffG41 = [6.0]
thetaRangeG41 = [1, 2, 4] 
inversionG41 = [1.0]

widthG42 = [0.001]      
cutoffG42 = [6.0]
thetaRangeG42 = [1, 2, 4] 
inversionG42 = [-1.0]

# angular part
"""legends2 = []
for width in widthG4:
    for cutoff in cutoffG4:
        for zeta in thetaRangeG4:   
            for inversion in inversionG4:
                functionValue = G4angular(theta, zeta, inversion)
                if plotTheta:
                    plt.plot(theta*180/np.pi, functionValue)
                else:
                    plt.plot(Rij, functionValue)
                legends2.append(r'$\zeta = %1.1f$' \
                               % zeta )
                plt.hold('on')
plt.legend(legends2)    
plt.show()"""

#plt.figure()

# complete function as function of theta
legends3 = []
for width in widthG41:
    for cutoff in cutoffG41:
        for zeta in thetaRangeG41:   
            for inversion in inversionG41:
                functionValue = G4(Rij, Rik, Rjk, theta, width, cutoff, zeta, inversion)
                if plotTheta:
                    plt.plot(theta*180/np.pi, functionValue)
                else:
                    plt.plot(Rij, functionValue)
                legends3.append(r'$\eta = %3.2f, R_c = %1.2f, \zeta = %1.1f, \lambda = %d$' \
                               % (width, cutoff, zeta, inversion) )
                plt.hold('on')
                
for width in widthG42:
    for cutoff in cutoffG42:
        for zeta in thetaRangeG42:   
            for inversion in inversionG42:
                functionValue = G4(Rij, Rik, Rjk, theta, width, cutoff, zeta, inversion)
                if plotTheta:
                    plt.plot(theta*180/np.pi, functionValue)
                else:
                    plt.plot(Rij, functionValue)
                legends3.append(r'$\eta = %3.2f, R_c = %1.2f, \zeta = %1.1f, \lambda = %d$' \
                               % (width, cutoff, zeta, inversion) )
                plt.hold('on')
                
plt.legend(legends3)    
plt.show()






