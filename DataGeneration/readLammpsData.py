# read data from LAMMPS runs for training

import numpy as np
import sys
np.random.seed(1)

def readXYZ(filename):
    
    inFile = open(filename, 'r')
    
    # skip three first lines
    for _ in xrange(3):
        inFile.readline()
        
    numberOfAtoms = int(inFile.readline())
    print "Number of atoms: ", numberOfAtoms
    
    inFile.readline()
    
    systemSize = []
    for _ in xrange(3):
        systemSize.append( float(inFile.readline().split()[1]) )
    systemSize = np.array(systemSize)
    systemSizeHalf = systemSize / 2.0
    print "System size: ", systemSize
    print "System size half: ", systemSizeHalf
    
    inFile.readline()
    
    # read file and stor positions in array
    positions = np.zeros((numberOfAtoms, 3))   
    counter = 0     
    for line in inFile:
        coordinates = line.split()
        positions[counter,0] = float(coordinates[0])
        positions[counter,1] = float(coordinates[1])
        positions[counter,2] = float(coordinates[2])
        counter += 1
        
    
    cut = 3.77
    x = []; y = []; z = []; r = []
    for i in xrange(numberOfAtoms):
        atom1 = positions[i]
        xNeigh = []; yNeigh = []; zNeigh = []; rNeigh = []
        for j in xrange(numberOfAtoms):
            if i != j:
                atom2 = positions[j]
                dr = atom1 - atom2 
                
                # periodic boundary conditions
                for dim in xrange(3):
                    if dr[dim] > systemSizeHalf[dim]:
                        dr[dim] -= systemSize[dim]
                    elif dr[dim] < -systemSizeHalf[dim]:
                        dr[dim] += systemSize[dim]
                    
                distance = np.sqrt( dr[0]**2 + dr[1]**2 + dr[2]**2 )
                
                # add to neighbour lists if r2 under cut
                if distance < cut:
                    xNeigh.append(dr[0])
                    yNeigh.append(dr[1])
                    zNeigh.append(dr[2])
                    rNeigh.append(distance)
                    
        x.append(xNeigh)
        y.append(yNeigh)
        z.append(zNeigh)
        r.append(rNeigh)
  
        # show progress
        sys.stdout.write("\r%2d %% complete" % ((float(i)/numberOfAtoms)*100))
        sys.stdout.flush()
    print r[1][:]

    return x, y, z, r
        
if __name__ == '__main__':
    readXYZ("../../LAMMPS_test/Silicon/Data/Si1000.xyz")