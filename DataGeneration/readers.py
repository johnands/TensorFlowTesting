import numpy as np
import tensorflow as tf
import sys


def readXYZ(filename, cut):
    """
    Make four nested lists x, y, z, r where vector x[i][:]
    is the x-coordinates of all the neighbours of atom i
    """
    
    # process xyz-file
    with open(filename, 'r') as inFile:
    
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
        
        # read positions and store in array
        positions = np.zeros((numberOfAtoms, 3))   
        counter = 0     
        for line in inFile:
            coordinates = line.split()
            positions[counter,0] = float(coordinates[0])
            positions[counter,1] = float(coordinates[1])
            positions[counter,2] = float(coordinates[2])
            counter += 1
        
    # make neighbour list for all atoms in the system
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

    return x, y, z, r
    
    
def readNeighbourData(filename):
    """
    Ordinary neighbour file:
    x1 y1 z1 r1 x2 y2 z2 r2 ... xN yN zN E
    """
    
    with open(filename, 'r') as inFile:
        
        x = []; y = []; z = []
        r = []; E = []
        for line in inFile:
            words = line.split()
            N = (len(words) - 1) / 4
            xi = []; yi = []; zi = [];
            ri = [];
            for i in xrange(N):
                xi.append(float(words[4*i]))
                yi.append(float(words[4*i+1]))
                zi.append(float(words[4*i+2]))
                ri.append(float(words[4*i+3]))
                
            x.append(xi)
            y.append(yi)
            z.append(zi)
            r.append(ri)
            E.append([float(words[-1])])  
                      
    return x, y, z, r, E
    
    
def readNeighbourDataMultiType(filename):
    """
    Ordinary neighbour file:
    x1 y1 z1 r1 type1 x2 y2 z2 r2 type2 ... xN yN zN typeN E
    """
    
    with open(filename, 'r') as inFile:
        
        x = []; y = []; z = []
        r = []; E = []
        types = []
        for line in inFile:
            words = line.split()
            N = (len(words) - 1) / 5
            xi = []; yi = []; zi = [];
            ri = [];
            typesi = [];
            for i in xrange(N):
                xi.append(float(words[5*i]))
                yi.append(float(words[5*i+1]))
                zi.append(float(words[5*i+2]))
                ri.append(float(words[5*i+3]))
                typesi.append(int(words[5*i+4]))
                
            x.append(xi)
            y.append(yi)
            z.append(zi)
            r.append(ri)
            types.append(typesi)
            E.append([float(words[-1])])  
                      
    return x, y, z, r, types, E
    
    
def readEnergy(filename):
    """
    Read energy only
    """
    
    with open(filename, 'r') as inFile:
        
        E = []
        for line in inFile:
            words = line.split()
            E.append([float(words[-1])])  
                      
    return E
    

def readNeighbourDataForce(filename):
    """
    Ordinary neighbour file:
    x1 y1 z1 r1 x2 y2 z2 r2 ... xN yN zN E Fx Fy Fz
    """
    
    with open(filename, 'r') as inFile:
        
        x = []; y = []; z = []; r = [];
        E = []; Fx = []; Fy = []; Fz = []
        for line in inFile:
            words = line.split()

            N = (len(words) - 4) / 4
            xi = []; yi = []; zi = [];
            ri = [];
            for i in xrange(N):
                xi.append(float(words[4*i]))
                yi.append(float(words[4*i+1]))
                zi.append(float(words[4*i+2]))
                ri.append(float(words[4*i+3]))
                
            x.append(xi)
            y.append(yi)
            z.append(zi)
            r.append(ri)
            E.append([float(words[-4])])  
            Fx.append([float(words[-3])])
            Fy.append([float(words[-2])])
            Fz.append([float(words[-1])])
            
    return x, y, z, r, E, Fx, Fy, Fz
    

def readNeighbourDataForceTag(filename):
    """
    Ordinary neighbour file:
    tag1 x1 y1 z1 r1 tag2 x2 y2 z2 r2 ... tagN xN yN zN E Fx Fy Fz
    """   
    
    with open(filename, 'r') as inFile:
        
        x = []; y = []; z = []; r = [];
        tags = []
        E = []; Fx = []; Fy = []; Fz = []
        for line in inFile:
            words = line.split()

            N = (len(words) - 4) / 5
            xi = []; yi = []; zi = [];
            tagsi = []
            ri = [];
            for i in xrange(N):
                tagsi.append(float(words[5*i]))
                xi.append(float(words[5*i+1]))
                yi.append(float(words[5*i+2]))
                zi.append(float(words[5*i+3]))
                ri.append(float(words[5*i+4]))
                
            tags.append(tagsi)
            x.append(xi)
            y.append(yi)
            z.append(zi)
            r.append(ri)
            E.append([float(words[-4])])  
            Fx.append([float(words[-3])])
            Fy.append([float(words[-2])])
            Fz.append([float(words[-1])])
            
    return x, y, z, r, E, Fx, Fy, Fz, tags
    
    
    
def readEnergyAndForce(filename):
    """
    Read energy and force only
    """
    
    with open(filename, 'r') as inFile:
        
        E = []; Fx = []; Fy = []; Fz = []
        for line in inFile:
            words = line.split()
            E.append([float(words[-4])])  
            Fx.append([float(words[-3])])
            Fy.append([float(words[-2])])
            Fz.append([float(words[-1])])            
            
    return E, Fx, Fy, Fz
    
    
def readSymmetryData(filename):
    
    inputData = []
    with open(filename, 'r') as infile:
        
        for line in infile:
            inputVector = []
            words = line.split()
            for word in words:
                inputVector.append(float(word))
            inputData.append(inputVector)
            
    return np.array(inputData)
    
    
def readParameters(filename):
    
    parameters = []
    with open(filename, 'r') as infile:
        infile.readline()
        for line in infile:
            param = []
            words = line.split()
            for word in words:
                param.append(float(word))
            parameters.append(param)
            
    return parameters
    
    
def readMetaFile(filename):
    
    # must first create a NN with the same architecture as the
    # on I want to load
    with open(filename, 'r') as infile:
        
        # read number of nodes and layers
        words = infile.readline().split()
        nNodes = int(words[-3][:-1])
        nLayers = int(words[-1])
        print "Number of nodes: ", nNodes
        print "Number of layers: ", nLayers
        
        # read activation function
        words = infile.readline().split()
        activation = words[5][:-1]
        print "Activation: ", activation
        if activation == 'sigmoid':
            activation = tf.nn.sigmoid
        elif activation == 'tanh':
            activation = tf.nn.tanh
        else:
            print activation, " is not a valid activation"
            
        # read inputs, outputs and lammps sample folder
        words = infile.readline().split()
        inputs = int(words[1][:-1])
        outputs = int(words[3][:-1])
        lammpsDir = words[-1]
        print "Inputs: ", inputs
        print "Outputs: ", outputs
        print "Lammps folder: ", lammpsDir
        
    return nNodes, nLayers, activation, inputs, outputs, lammpsDir
        
        
def readSymmFunc(filename):
    
    with open(filename, 'r') as infile:
        
        infile.readline()
        
        words = infile.readline().split()
        symmFuncType = words[-1]
        print "Symmetry function type: ", symmFuncType
    
    return symmFuncType
        
        
        
        
        
        