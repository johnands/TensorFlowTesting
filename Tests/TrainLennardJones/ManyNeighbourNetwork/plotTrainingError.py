"""
Plot the training error when training for 20 neighbours, 1 layer,
200 nodes
"""

import numpy as np
import matplotlib.pyplot as plt

inFile = open('../../../TrainingData/02.12-19.36.50/meta.dat', 'r')
#inFile = open('../../../TrainingData/13.12-12.59.51/meta.dat', 'r')

# skip comment lines
inFile.readline()
inFile.readline()
inFile.readline()

epoch = []
trainCost = []
testCost = []
for line in inFile:
    words = line.split()
    epoch.append(float(words[0]))
    trainCost.append(float(words[1]))
    testCost.append(float(words[2]))

epoch = np.array(epoch)
trainCost = np.array(trainCost)
testCost = np.array(testCost)

cut = int(0.3*len(epoch))
cut = -1

plt.plot(epoch[:cut], trainCost[:cut], 'b-', epoch[:cut], testCost[:cut], 'g-')
plt.legend(['Training set cost', 'Test set cost'], fontsize=15)
plt.axis([0, 5e4, 0, np.max(trainCost)])
plt.xlabel('Epoch', fontsize=15)
plt.ylabel('Quadratic cost', fontsize=15)
plt.show()
#plt.savefig('Plots/trainingManyNeighbourNN1.pdf')

plt.subplot(2,1,1)
plt.plot(epoch[:cut], trainCost[:cut], 'b-', epoch[:cut], testCost[:cut], 'g-')
plt.legend(['Training set cost', 'Test set cost'], fontsize=15)
plt.axis([0, 16282000, 0, 0.1])
plt.ylabel('Quadratic cost', fontsize=15)

plt.subplot(2,1,2)
plt.plot(epoch[:cut], trainCost[:cut], 'b-', epoch[:cut], testCost[:cut], 'g-')
plt.legend(['Training set cost', 'Test set cost'], loc=3, fontsize=15)
plt.axis([0, 16282000, 0, 0.001])
plt.xlabel('Epoch', fontsize=15)
plt.ylabel('Quadratic cost', fontsize=15)
plt.show()
#plt.savefig('Plots/trainingManyNeighbourNN2.pdf')