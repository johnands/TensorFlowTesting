""""
Visualize symmetry functions used in a specific training session
"""


import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
import DataGeneration.readers as readers
import sys

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
           
def G4G5angular(theta, zeta, inversion):
    
    return 2**(1-zeta) * (1 + inversion*np.cos(theta))**zeta
           
           
def G5(Rij, Rik, cosTheta, width, cutoff, thetaRange, inversion):
    
    return 2**(1-thetaRange) * (1 + inversion*cosTheta)**thetaRange * \
           np.exp( -width*(Rij**2 + Rik**2) ) * \
           cutoffFunction(Rij, cutoff) * cutoffFunction(Rik, cutoff)

# set parameters
plt.rc('lines', linewidth=1.5)
#plt.rc('axes', prop_cycle=(cycler('color', ['g', 'k', 'y', 'b', 'r', 'c', 'm']) ))
plt.rc('xtick', labelsize=20)
plt.rc('ytick', labelsize=20)
plt.rc('axes', labelsize=25)

# parse arguments
filename = sys.argv[1]

saveFlag = False
saveFileName = ''
if len(sys.argv) > 2:
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == '--save':
            i += 1
            saveFlag     = True
            saveFileName = sys.argv[i]
            i += 1
            break

# read parameters
parameters = readers.readParameters('Parameters/' + filename)

# read symmetry function type
#symmFuncType = readers.readSymmFunc(trainingDir + '/meta.dat')
symmFuncType = 'G5'

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
Rij3 = np.linspace(0, globalCutoff + 2, 1000)


##### G2 plot #####

legends = []
for eta, Rc, Rs in parameters2:
    functionValue = G2(Rij2, eta, Rc, Rs)
    functionValue[np.where(Rij2 > Rc)[0]] = 0
    plt.plot(Rij2, functionValue)
    legends.append(r'$\eta=%3.2f \, \mathrm{\AA{}}^{-2}, R_c=%1.1f  \, \mathrm{\AA{}}, R_s=%1.1f \, \mathrm{\AA{}}$' % \
                   (eta, Rc, Rs) )
    plt.hold('on')

plt.legend(legends, prop={'size':20})
plt.ylabel(r'$G_2, \, R_s = 0 \, \mathrm{\AA{}}$')
plt.tight_layout()
plt.show()
if saveFlag:
    plt.savefig('../Figures/Theory/G2_1.pdf')


##### G4/G5 plot #####

"""plt.figure()

theta = np.linspace(0, 2*np.pi, 1000) 

inversion = 1.0
legends = []
for zeta in [1.0, 2.0, 4.0, 16.0, 64.0]:
    functionValue = G5(theta, zeta, inversion)
    plt.plot(theta*180/np.pi, functionValue)
    legends.append(r'$\zeta = %d$' % zeta)
    plt.hold('on')
    
plt.legend(legends, prop={'size':20}, loc=9)
plt.xlabel(r'$\theta$')
plt.ylabel(r'$G^4/G^5$ angular part')
plt.axis([0, 2*180, 0, 2])
plt.tight_layout()
plt.show()
#plt.savefig('../Figures/Theory/G4G5angular1.pdf')
    
plt.figure()    
    
inversion = -1.0
legends = []
for zeta in [1.0, 2.0, 4.0, 16.0, 64.0]:
    functionValue = G5(theta, zeta, inversion)
    plt.plot(theta*180/np.pi, functionValue)
    legends.append(r'$\zeta = %d$' % zeta)
    plt.hold('on')

plt.legend(legends, fontsize=25, prop={'size':18}, loc=1)
plt.xlabel(r'$\theta$')
plt.ylabel(r'$G^4/G^5$ angular part')
plt.axis([0, 2*180, 0, 2])
plt.tight_layout()
plt.show()
#plt.savefig('../Figures/Theory/G4G5angular2.pdf')
"""



