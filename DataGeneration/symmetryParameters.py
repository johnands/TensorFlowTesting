# tweak parameters and visualize

# define my own types:
# type 1: G2 with Rs = 0
# type 2: G2 with Rs != 0
# type 3: G4 with varied cutoffs
# type 4: G4 with varied zetas


def writeParameters(parameters, filename):
    """Write parameter set to file for later use"""
    
    numberOfParameters = len(parameters)

    with open(filename, 'w') as outfile:
        
        # write number of symmfuncs and number of unique parameters
        outStr = "%d" % (numberOfParameters)
        outfile.write(outStr + '\n')
        for symmFunc in parameters:
            for param in symmFunc:
                outfile.write("%g" % param)
                if numberOfParameters > 1:
                    outfile.write(" ")
            outfile.write("\n")


def SiBehler(write=False):
    """
    The Si parameters we received from Behler. 
    Behler has defined different kind of types
    """   
    
    # make nested list of all symmetry function parameters
    parameters = []    
    
    # type1
    center = 0.0
    cutoff = 6.0
    for eta in [2.0, 0.5, 0.2, 0.1, 0.04, 0.001]:
        parameters.append([eta, cutoff, center])
    
    # type2
    zeta = 1.0
    inversion = 1.0
    eta = 0.01
    for cutoff in [6.0, 5.5, 5.0, 4.5, 4.0, 3.5]:
        parameters.append([eta, cutoff, zeta, inversion])
        
    # type 3
    cutoff = 6.0
    eta = 4.0
    for center in [5.5, 5.0, 4.5, 4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0]:
        parameters.append([eta, cutoff, center])
        
        
    eta = 0.01
    
    # type 4
    zeta = 1.0
    inversion = -1.0    
    for cutoff in [6.0, 5.5, 5.0, 4.5, 4.0, 3.5]:
        parameters.append([eta, cutoff, zeta, inversion])
        
    # type 5 and 6
    zeta = 2.0
    for inversion in [1.0, -1.0]:
        for cutoff in [6.0, 5.0, 4.0, 3.0]:
            parameters.append([eta, cutoff, zeta, inversion])
        
    # type 7 and 8
    zeta = 4.0
    for inversion in [1.0, -1.0]:
        for cutoff in [6.0, 5.0, 4.0, 3.0]:
            parameters.append([eta, cutoff, zeta, inversion])
    
    # type 9 and 10
    zeta = 16.0
    for inversion in [1.0, -1.0]:
        for cutoff in [6.0, 4.0]:
            parameters.append([eta, cutoff, zeta, inversion])
            
    if write:
        writeParameters(parameters, 'Parameters/SiBehler.dat')
        
    return parameters
    
    
def SiBulkCustom(write=False):
    
    # make nested list of all symmetry function parameters
    # parameters from Behler
    parameters = []    
    
    # type1
    center = 0.0
    cutoff = 3.77118
    for eta in [2.0, 0.5, 0.2, 0.1, 0.04, 0.001]:
        parameters.append([eta, cutoff, center])
    
    # type2
    zeta = 1.0
    inversion = 1.0
    eta = 0.01
    for cutoff in [6.0, 5.5, 5.0, 4.5, 4.0, 3.8]:
        parameters.append([eta, cutoff, zeta, inversion])
        
    # type 3
    cutoff = 6.0
    eta = 4.0
    for center in [5.5, 5.0, 4.5, 4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0]:
        parameters.append([eta, cutoff, center])
        
        
    eta = 0.01
    
    # type 4
    zeta = 1.0
    inversion = -1.0    
    for cutoff in [6.0, 5.5, 5.0, 4.5, 4.0, 3.8]:
        parameters.append([eta, cutoff, zeta, inversion])
        
    # type 5 and 6
    zeta = 2.0
    for inversion in [1.0, -1.0]:
        for cutoff in [6.0, 5.0, 4.0, 3.8]:
            parameters.append([eta, cutoff, zeta, inversion])
        
    # type 7 and 8
    zeta = 4.0
    for inversion in [1.0, -1.0]:
        for cutoff in [6.0, 5.0, 4.0, 3.8]:
            parameters.append([eta, cutoff, zeta, inversion])
    
    # type 9 and 10
    zeta = 16.0
    for inversion in [1.0, -1.0]:
        for cutoff in [6.0, 4.0]:
            parameters.append([eta, cutoff, zeta, inversion])  
            
    return parameters
        
        
def Si3Atoms(write):
    """Parameter set customized for a system of 3 atoms"""    
    
    parameters = []
    
    ##### G2 #####
    
    # global cutoff
    globalCut = 3.77118
    
    # type 1
    Rs = 0  
    for eta in [2.0, 1.0, 0.1, 0.01]:
        parameters.append([eta, globalCut, Rs])
        
    # type 2
    eta = 4.0
    for Rs in [3.5, 3.0, 2.5, 2.0]:
        parameters.append([eta, globalCut, Rs])
        
        
    ##### G5 #####
    
    eta = 0.01
    
    # type 3
    zeta = 1
    for Lambda in [1, -1]:
        for Rc in [globalCut, 3.5, 2.5]:
            parameters.append([eta, Rc, zeta, Lambda])
        
    # type 3
    zeta = 4.0
    for Lambda in [1, -1]:
        for Rc in [globalCut, 3.5, 2.5]:
            parameters.append([eta, Rc, zeta, Lambda])
            
    if write:
        writeParameters(parameters, 'Parameters/Si3atoms.dat')
        
    return parameters
    
    
def SiO2type0(write=False):
    
    # make nested list of all symetry function parameters
    parameters = []  
    
    # G2
    # Si-Si, Si-O customize later
    # general trend pairs: 
    # number of pair neighbours: 51-62
    # Si have ca 25 Si and 35 O as neighbours
    
    elem2param = {}
    i = 0
    iOld = 0  
    
    # Si-Si: [0,0]
    cutoff = 5.5
    
    center = 0.0  
    for eta in [2.0, 0.5, 0.2, 0.1, 0.04, 0.001]:
        parameters.append([eta, cutoff, center])
        i += 1
        
    eta = 4.0
    for center in [5.0, 4.5, 4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0, 0.5]:
        parameters.append([eta, cutoff, center])
        i += 1
        
    iNew = i
    elem2param[(0,0)] = (iOld, iNew)
    iOld = i
        
    # Si-O: [0,1]
    cutoff = 5.5
    
    center = 0.0  
    for eta in [2.0, 0.5, 0.2, 0.1, 0.04, 0.001]:
        parameters.append([eta, cutoff, center])
        i += 1
        
    eta = 4.0
    for center in [5.0, 4.5, 4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0, 0.5]:
        parameters.append([eta, cutoff, center])
        i += 1
        
    iNew = i
    elem2param[(0,1)] = (iOld, iNew)
    iOld = i
           
    # G5
    # general trend triplets:
    # number of triplet neighbours: 2-8
           
    # Si-Si-Si: [0,0,0]
    """eta = 0.01
    
    zeta = 1.0
    inversion = 1.0
    for cutoff in [3.5, 3.0, 2.5]:
        for inversion in [-1.0, 1.0]:
            parameters.append([eta, cutoff, zeta, inversion])  
            i += 1
            
    zeta = 4.0
    inversion = 1.0
    for cutoff in [3.5, 3.0, 2.5]:
        for inversion in [-1.0, 1.0]:
            parameters.append([eta, cutoff, zeta, inversion])  
            i += 1
            
    iNew = i
    elem2param[(0,0,0)] = (iOld, iNew)
    iOld = i
            
    # Si-Si-O: [0,0,1]
    eta = 0.01
    
    zeta = 1.0
    inversion = 1.0
    for cutoff in [3.5, 3.0, 2.5]:
        for inversion in [-1.0, 1.0]:
            parameters.append([eta, cutoff, zeta, inversion])  
            i += 1
            
    zeta = 4.0
    inversion = 1.0
    for cutoff in [3.5, 3.0, 2.5]:
        for inversion in [-1.0, 1.0]:
            parameters.append([eta, cutoff, zeta, inversion])
            i += 1
            
    iNew = i
    elem2param[(0,0,1)] = (iOld, iNew)
    iOld = i
            
    # Si-O-Si: [0,1,0]
    eta = 0.01
    
    zeta = 1.0
    inversion = 1.0
    for cutoff in [3.5, 3.0, 2.5]:
        for inversion in [-1.0, 1.0]:
            parameters.append([eta, cutoff, zeta, inversion])  
            i += 1
            
    zeta = 4.0
    inversion = 1.0
    for cutoff in [3.5, 3.0, 2.5]:
        for inversion in [-1.0, 1.0]:
            parameters.append([eta, cutoff, zeta, inversion])  
            i += 1
            
    iNew = i
    elem2param[(0,1,0)] = (iOld, iNew)
    iOld = i"""
    
    # Si-O-O: [0,1,1]
    eta = 0.01
    
    zeta = 1.0
    inversion = 1.0
    for cutoff in [3.5, 3.0, 2.5]:
        for inversion in [-1.0, 1.0]:
            parameters.append([eta, cutoff, zeta, inversion]) 
            i += 1
            
    zeta = 4.0
    inversion = 1.0
    for cutoff in [3.5, 3.0, 2.5]:
        for inversion in [-1.0, 1.0]:
            parameters.append([eta, cutoff, zeta, inversion]) 
            i += 1
        
    iNew = i
    elem2param[(0,1,1)] = (iOld, iNew)
    iOld = i
            
    return parameters, elem2param

        
        
def SiO2type1(write=False):
    
    # make nested list of all symetry function parameters
    # parameters from Behler
    parameters = []   
    
    # G2
    # O-O and O-Si: customize later
    # general trend pairs: 
    # number of pair neighbours: 51-62
    # O  have ca 20 Si and 40 O as neighbours 
    
    elem2param = {}
    i = 0
    iOld = 0  
    
    # O-Si: [1,0]
    cutoff = 5.5
    
    center = 0.0  
    for eta in [2.0, 0.5, 0.2, 0.1, 0.04, 0.001]:
        parameters.append([eta, cutoff, center])
        i += 1
        
    eta = 4.0
    for center in [5.0, 4.5, 4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0, 0.5]:
        parameters.append([eta, cutoff, center])
        i += 1
        
    iNew = i
    elem2param[(1,0)] = (iOld, iNew)
    iOld = i
        
    # O-O: [1,1]
    cutoff = 5.5
    
    center = 0.0  
    for eta in [2.0, 0.5, 0.2, 0.1, 0.04, 0.001]:
        parameters.append([eta, cutoff, center])
        i += 1
        
    eta = 4.0
    for center in [5.0, 4.5, 4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0, 0.5]:
        parameters.append([eta, cutoff, center])
        i += 1
        
    iNew = i
    elem2param[(1,1)] = (iOld, iNew)
    iOld = i
           
    # G5
    # general trend triplets:
    # number of triplet neighbours: 2-8
           
    # O-Si-Si: [1,0,0]
    eta = 0.01
    
    zeta = 1.0
    inversion = 1.0
    for cutoff in [3.5, 3.0, 2.5]:
        for inversion in [-1.0, 1.0]:
            parameters.append([eta, cutoff, zeta, inversion])  
            i += 1
            
    zeta = 4.0
    inversion = 1.0
    for cutoff in [3.5, 3.0, 2.5]:
        for inversion in [-1.0, 1.0]:
            parameters.append([eta, cutoff, zeta, inversion])  
            i += 1
            
    iNew = i
    elem2param[(1,0,0)] = (iOld, iNew)
    iOld = i
            
    # O-Si-O: [1,0,1]
    """eta = 0.01
    
    zeta = 1.0
    inversion = 1.0
    for cutoff in [3.5, 3.0, 2.5]:
        for inversion in [-1.0, 1.0]:
            parameters.append([eta, cutoff, zeta, inversion])  
            i += 1
            
    zeta = 4.0
    inversion = 1.0
    for cutoff in [3.5, 3.0, 2.5]:
        for inversion in [-1.0, 1.0]:
            parameters.append([eta, cutoff, zeta, inversion])
            i += 1
            
    iNew = i
    elem2param[(1,0,1)] = (iOld, iNew)
    iOld = i
            
    # O-O-Si: [1,1,0]
    eta = 0.01
    
    zeta = 1.0
    inversion = 1.0
    for cutoff in [3.5, 3.0, 2.5]:
        for inversion in [-1.0, 1.0]:
            parameters.append([eta, cutoff, zeta, inversion])  
            i += 1
            
    zeta = 4.0
    inversion = 1.0
    for cutoff in [3.5, 3.0, 2.5]:
        for inversion in [-1.0, 1.0]:
            parameters.append([eta, cutoff, zeta, inversion])  
            i += 1
            
    iNew = i
    elem2param[(1,1,0)] = (iOld, iNew)
    iOld = i
    
    # O-O-O: [1,1,1]
    eta = 0.01
    
    zeta = 1.0
    inversion = 1.0
    for cutoff in [3.5, 3.0, 2.5]:
        for inversion in [-1.0, 1.0]:
            parameters.append([eta, cutoff, zeta, inversion]) 
            i += 1
            
    zeta = 4.0
    inversion = 1.0
    for cutoff in [3.5, 3.0, 2.5]:
        for inversion in [-1.0, 1.0]:
            parameters.append([eta, cutoff, zeta, inversion]) 
            i += 1
        
    iNew = i
    elem2param[(1,1,1)] = (iOld, iNew)
    iOld = i"""
            
    return parameters, elem2param
    
    
def SiO2atoms2type0(write=False):
    
    # make nested list of all symetry function parameters
    parameters = []  
    
    # G2
    # Si-Si, Si-O customize later
    # general trend pairs: 
    # number of pair neighbours: 51-62
    # Si have ca 25 Si and 35 O as neighbours
    
    elem2param = {}
    i = 0
    iOld = 0  
    
    # Si-Si: [0,0]
    """cutoff = 5.5
    
    center = 0.0  
    for eta in [2.0, 0.5, 0.2, 0.1, 0.04, 0.001]:
        parameters.append([eta, cutoff, center])
        i += 1
        
    eta = 4.0
    for center in [5.0, 4.5, 4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0, 0.5]:
        parameters.append([eta, cutoff, center])
        i += 1
        
    iNew = i
    elem2param[(0,0)] = (iOld, iNew)
    iOld = i"""
        
    # Si-O: [0,1]
    cutoff = 5.5
    
    center = 0.0  
    for eta in [2.0, 0.5, 0.2, 0.1, 0.04, 0.001]:
        parameters.append([eta, cutoff, center])
        i += 1
        
    eta = 4.0
    for center in [5.0, 4.5, 4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0, 0.5]:
        parameters.append([eta, cutoff, center])
        i += 1
        
    iNew = i
    elem2param[(0,1)] = (iOld, iNew)
    iOld = i
    
    if write:
        writeParameters(parameters, '../Parameters/SiO2atoms2.dat')
    
    return parameters, elem2param
    

def SiO2atoms2type1(write=False):
    
    # make nested list of all symetry function parameters
    # parameters from Behler
    parameters = []   
    
    # G2
    # O-O and O-Si: customize later
    # general trend pairs: 
    # number of pair neighbours: 51-62
    # O  have ca 20 Si and 40 O as neighbours 
    
    elem2param = {}
    i = 0
    iOld = 0  
    
    # O-Si: [1,0]
    cutoff = 5.5
    
    center = 0.0  
    for eta in [2.0, 0.5, 0.2, 0.1, 0.04, 0.001]:
        parameters.append([eta, cutoff, center])
        i += 1
        
    eta = 4.0
    for center in [5.0, 4.5, 4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0, 0.5]:
        parameters.append([eta, cutoff, center])
        i += 1
        
    iNew = i
    elem2param[(1,0)] = (iOld, iNew)
    iOld = i
        
    # O-O: [1,1]
    """cutoff = 5.5
    
    center = 0.0  
    for eta in [2.0, 0.5, 0.2, 0.1, 0.04, 0.001]:
        parameters.append([eta, cutoff, center])
        i += 1
        
    eta = 4.0
    for center in [5.0, 4.5, 4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0, 0.5]:
        parameters.append([eta, cutoff, center])
        i += 1
        
    iNew = i
    elem2param[(1,1)] = (iOld, iNew)
    iOld = i"""
    
    if write:
        writeParameters(parameters, '../Parameters/SiO2atoms2.dat')
    
    return parameters, elem2param
       

        
    
if __name__ == '__main__':
        # make nested list of all symetry function parameters
    # parameters from Behler
    parameters = []   
    
    # G2
    # O-O and O-Si: customize later
    # general trend pairs: 
    # number of pair neighbours: 51-62
    # O  have ca 20 Si and 40 O as neighbours 
    
    elem2param = {}
    i = 0
    iOld = 0  
    
    # O-Si: [1,0]
    cutoff = 5.5
    
    center = 0.0  
    for eta in [2.0, 0.5, 0.2, 0.1, 0.04, 0.001]:
        parameters.append([eta, cutoff, center])
        i += 1
        
    eta = 4.0
    for center in [5.0, 4.5, 4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0, 0.5]:
        parameters.append([eta, cutoff, center])
        i += 1
        
    iNew = i
    elem2param[(1,0)] = (iOld, iNew)
    iOld = i
        
    # O-O: [1,1]
    cutoff = 5.5
    
    center = 0.0  
    for eta in [2.0, 0.5, 0.2, 0.1, 0.04, 0.001]:
        parameters.append([eta, cutoff, center])
        i += 1
        
    eta = 4.0
    for center in [5.0, 4.5, 4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0, 0.5]:
        parameters.append([eta, cutoff, center])
        i += 1
        
    iNew = i
    elem2param[(1,1)] = (iOld, iNew)
    iOld = i
    #SiBehler(write=False)
    #Si3Atoms(write=True)
    SiO2type02atoms(write=True)
        
        

    

    
    
            









