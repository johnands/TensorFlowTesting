import matplotlib.pyplot as plt
import numpy as np

# plot RMSE as function of epoch

def readError(filename):
    
    with open(filename) as infile:
        
        # skip headers
        infile.readline(); infile.readline(); infile.readline()
        
        # read RMSE of train and test
        epoch = []; trainError = []; testError = [];
        for line in infile:
            words = line.split()
            epoch.append(float(words[0]))
            trainError.append(float(words[1]))
            testError.append(float(words[2]))
    
    return epoch, trainError, testError
    
    
file1 = '../TrainingData/TestOverfitting/NoOverfit/meta.dat'    # no overfit
file2 = '../TrainingData/TestOverfitting/Overfit/meta.dat'      # overfit

epoch1, trainError1, testError1 = readError(file1)
epoch2, trainError2, testError2 = readError(file2)

assert np.array_equal(epoch1, epoch2)

fig = plt.figure()

ax = fig.add_subplot(2,1,1)
plt.plot(epoch1, trainError1, 'b-', epoch1, testError1, 'g-')
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.legend(['Training set', 'Test set'], prop={'size':15})
ax.text(0.05, 0.8, 'a)', fontsize=20,
        #horizontalalignment='left',
        transform=ax.transAxes)
plt.axis([0, 15000, 0, 0.01])

ax = fig.add_subplot(2,1,2)       
plt.plot(epoch2, trainError2, 'b-', epoch2, testError2, 'g-')
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.legend(['Training set', 'Test set'], prop={'size':15})
ax.text(0.05, 0.8, 'b)', fontsize=20,
        #horizontalalignment='left',
        transform=ax.transAxes)
plt.axis([0, 15000, 0, 0.01])

plt.tight_layout()
plt.savefig('../../Oppgaven/Figures/Implementation/overfitting.pdf')
#plt.show()