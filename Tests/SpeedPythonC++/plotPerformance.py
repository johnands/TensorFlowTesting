import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt

# load files
files = []
files.append(open("evaluateNetworkTest.dat", 'r'))
files.append(open("speedC++.dat", 'r'))
files.append(open("speedArmadillo.dat", 'r'))

# skip comment lines
for fileName in files:
    fileName.readline()
    
time = []
networkSize = []
nodes = []
layers = []
for i in range(len(files)):
    time.append([])
    for line in files[i]:
        words = line.split()
        if i == 0:
            nLayers = (int(words[0]))
            nNodes = (int(words[1]))
            nodes.append(nNodes)
            layers.append(nLayers)
            networkSize.append(nLayers*nNodes)
        time[i].append(float(words[2]))

# convert to arrays    
layers = np.array(layers)
nodes = np.array(nodes)
networkSize = np.array(networkSize)
time = np.array(time)


##### plots #####

# plot for specific number of nodes
#plt.figure()
#nodeNumber = 10
#indicies = np.where(nodes == nodeNumber)
#layersN100 = layers[indicies]
#timePython = time[0][indicies]
#timeTF = time[1][indicies]
#timeArma = time[2][indicies]
#plt.subplot(3,1,1)
#plt.plot(layersN100, timeTF/timePython)
#plt.xlabel('Number of layers')
#plt.ylabel('Time TFC++ / time TFpython')
#plt.subplot(3,1,2)
#plt.plot(layersN100, timeTF/timeArma)
#plt.xlabel('Number of layers')
#plt.ylabel('Time TFC++ / time armadillo')
#plt.subplot(3,1,3)
#plt.plot(layersN100, timePython/timeArma)
#plt.xlabel('Number of layers')
#plt.ylabel('Time TFpython / time armadillo')
#plt.show()

# plot for specific number of layeres
#plt.figure()
#layerNumber = 5
#cut = 0
#indicies = np.where(layers == layerNumber)
#nodesL30 = nodes[indicies]
#timePython = time[0][indicies]
#timeTF = time[1][indicies]
#timeArma = time[2][indicies]
#plt.subplot(3,1,1)
#plt.plot(nodesL30, timeTF/timePython)
#plt.xlabel('Number of nodes')
#plt.ylabel('Time TFC++ / time TFpython')
#plt.subplot(3,1,2)
#plt.plot(nodesL30[cut:], timeTF[cut:]/timeArma[cut:])
#plt.xlabel('Number of nodes')
#plt.ylabel('Time TFC++ / time armadillo')
#plt.subplot(3,1,3)
#plt.plot(nodesL30[cut:], timePython[cut:]/timeArma[cut:])
#plt.xlabel('Number of nodes')
#plt.ylabel('Time TFpython / time armadillo')
#plt.show()

# scatter plots
plt.figure()
plt.subplot(3,1,1)
plt.scatter(networkSize, time[1]/time[0])
plt.hold('on')
plt.plot(networkSize, np.zeros(len(networkSize)) + 1, 'r-', linewidth=1)
plt.xlabel(r'System size: $L\cdot N$', fontsize=15)
plt.ylabel(r'$T_{TFC} / T_{TFP}$', fontsize=15)
#plt.axis([0, 3100, 0, 2])

plt.subplot(3,1,2)
plt.scatter(networkSize, time[1]/time[2])
plt.hold('on')
plt.plot(networkSize, np.zeros(len(networkSize)) + 1, 'r-', linewidth=1)
plt.xlabel(r'System size: $L\cdot N$', fontsize=15)
plt.ylabel(r'$T_{TFC} / T_{ARMA}$', fontsize=15)
#plt.axis([0, 3100, 0, 10])

plt.subplot(3,1,3)
plt.scatter(networkSize, time[0]/time[2])
plt.hold('on')
plt.plot(networkSize, np.zeros(len(networkSize)) + 1, 'r-', linewidth=1)
#plt.axis([0, 3100, 0, 10])
plt.xlabel(r'System size: $L\cdot N$', fontsize=15)
plt.ylabel(r'$T_{TFP} / T_{ARMA}$', fontsize=15)
plt.suptitle('Comparison of time usage for different platforms', fontsize=15)
plt.savefig('timeComparisonNetwork2.pdf')
plt.show()



