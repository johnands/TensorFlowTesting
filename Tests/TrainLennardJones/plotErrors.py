import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

inFile = open('errorEnergyDerivativeC.dat', 'r')

energies = []
derivatives = []
for line in inFile:
    words = line.split()
    energies.append(float(words[0]))
    derivatives.append(float(words[1]))
    
cutoff = 2.5
offSet = 1.0/cutoff**12 - 1.0/cutoff**6
shiftedLJ = lambda s : 4*(1.0/s**12 - 1.0/s**6 - offSet)
analyticalDerivative = lambda t: 24*(t**6 - 2) / t**13

N = 1000
distances = np.linspace(0.8, 2.5, N)
energies = np.array(energies)
error = energies - shiftedLJ(distances)
#plt.plot(distances, error, 'g-')
#plt.legend(['NN(r) - LJ(r)'])
#plt.xlabel('r [MD]')
#plt.ylabel('E [MD]')
#plt.savefig('Plots/errorLJC.pdf')

derivatives = np.array(derivatives)
derivativesError = derivatives - analyticalDerivative(distances)
cut = int(10*1000/100)
plt.plot(distances[:cut], derivativesError[:cut], 'b-')
plt.legend([r'$NN^\prime(r) - LJ^\prime(r)$'], fontsize=15)
plt.xlabel('r [MD]', fontsize=15)
plt.ylabel('dE/dr [MD]', fontsize=15)
plt.savefig('Plots/errorDerivative.pdf')
#plt.show()
