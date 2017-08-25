import numpy as np
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
#nodeNumber = 100
#indicies = np.where(nodes == nodeNumber)
#layersN100 = layers[indicies]
#timePython = time[0][indicies]
#timeTF = time[1][indicies]
#timeArma = time[2][indicies]
#
#layerNumber = 30
#cut = 0
#indicies2 = np.where(layers == layerNumber)
#nodesL30 = nodes[indicies2]
#timePython2 = time[0][indicies2]
#timeTF2 = time[1][indicies2]
#timeArma2 = time[2][indicies2]
#
#plt.subplot(3,2,1)
#plt.plot(layersN100, timeTF/timePython)
#plt.hold('on')
#plt.plot(layersN100, np.zeros(len(layersN100)) + 1, 'r-', linewidth=1)
#plt.ylabel(r'$T_{TFC} / T_{TFP}$', fontsize=15)
#
#plt.subplot(3,2,2)
#plt.plot(nodesL30, timeTF2/timePython2)
#plt.hold('on')
#plt.plot(nodesL30, np.zeros(len(nodesL30)) + 1, 'r-', linewidth=1)
#
#plt.subplot(3,2,3)
#plt.plot(layersN100, timeTF/timeArma)
#plt.hold('on')
#plt.plot(layersN100, np.zeros(len(layersN100)) + 1, 'r-', linewidth=1)
#plt.axis([0, 30, 0, 10])
#plt.ylabel(r'$T_{TFC} / T_{ARMA}$', fontsize=15)
#
#plt.subplot(3,2,4)
#plt.plot(nodesL30[cut:], timeTF2[cut:]/timeArma2[cut:])
#plt.hold('on')
#plt.plot(nodesL30[cut:], np.zeros(len(nodesL30[cut:])) + 1, 'r-', linewidth=1)
#plt.axis([0, 100, 0, 10])
#
#plt.subplot(3,2,5)
#plt.plot(layersN100, timePython/timeArma)
#plt.hold('on')
#plt.plot(layersN100, np.zeros(len(layersN100)) + 1, 'r-', linewidth=1)
#plt.axis([0, 30, 0, 10])
#plt.xlabel('L', fontsize=15)
#plt.ylabel(r'$T_{TFP} / T_{ARMA}$', fontsize=15)
#
#plt.subplot(3,2,6)
#plt.plot(nodesL30[cut:], timePython2[cut:]/timeArma2[cut:])
#plt.hold('on')
#plt.plot(nodesL30, np.zeros(len(nodesL30)) + 1, 'r-', linewidth=1)
#plt.axis([0, 100, 0, 10])
#plt.xlabel('N', fontsize=15)
#plt.tight_layout()
#plt.suptitle('N = 100                                           ' + \
#              '                                                 ' + 
#              '                                     L = 30', fontsize=15)
#plt.savefig('Plots/timeComparisonNetwork3.pdf')
#plt.show()


# scatter plots

# convert to array of arrays
time = np.array([np.array(ti) for ti in time])

fig = plt.figure()

ax = fig.add_subplot(2,1,1)
plt.scatter(networkSize, time[1]/time[2])
plt.hold('on')
plt.plot(networkSize, np.zeros(len(networkSize)) + 1, 'r-', linewidth=1)
#plt.xlabel(r'System size: $L\cdot N$', fontsize=15)
plt.ylabel(r'$T_{TFC} / T_{ARMA}$', fontsize=15)
plt.axis([0, 200, 0, 310])
#ax.text(0.8, 0.8, 'a)', fontsize=18,
#        #horizontalalignment='left',
#        transform=ax.transAxes)

ax = fig.add_subplot(2,2,2)
plt.scatter(networkSize, time[1]/time[2])
plt.hold('on')
plt.plot(networkSize, np.zeros(len(networkSize)) + 1, 'r-', linewidth=1)
plt.axis([0, 200, 0, 50])
plt.xlabel(r'System size: $L\cdot N$', fontsize=15)
plt.ylabel(r'$T_{TFC} / T_{ARMA}$', fontsize=15)
ax.text(0.05, 0.2, 'b)', fontsize=18,
        #horizontalalignment='left',
        transform=ax.transAxes)
#plt.savefig('Plots/timeComparisonNetwork2.pdf')
#plt.show()

# full scatter plot for TFC++/Arma
#plt.figure()
ax = fig.add_subplot(2,2,3)
plt.scatter(networkSize, time[1]/time[2])
plt.hold('on')
plt.plot(networkSize, np.zeros(len(networkSize)) + 1, 'r-', linewidth=1)
plt.xlabel(r'System size: $L\cdot N$', fontsize=15)
plt.ylabel(r'$T_{TFC} / T_{ARMA}$', fontsize=15)
plt.axis([0, 3000, 0, 10])
ax.text(0.8, 0.8, 'c)', fontsize=18,
        #horizontalalignment='left',
        transform=ax.transAxes)


ax = fig.add_subplot(2,2,4)
print time[0].shape
plt.scatter(networkSize, time[1]/time[0])
plt.hold('on')
plt.plot(networkSize, np.zeros(len(networkSize)) + 1, 'r-', linewidth=1)
#plt.xlabel(r'System size: $L\cdot N$', fontsize=15)
plt.ylabel(r'$T_{TFC} / T_{TFP}$', fontsize=15)
plt.axis([0, 3000, 0, 3])
ax.text(0.05, 0.8, 'd)', fontsize=18,
        #horizontalalignment='left',
        transform=ax.transAxes)
plt.tight_layout()
plt.show()
#plt.savefig('../../../Oppgaven/Figures/Tests/timeComparisonNetworkNew.pdf')

