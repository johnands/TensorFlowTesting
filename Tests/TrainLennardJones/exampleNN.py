import tensorflow as tf
import numpy as np

allWeights = []
allBiases = []

nLayers = 2
nNodes = 2

weights1 = tf.Variable([0.6, 0.8])
biases1 = tf.Variable([0.1, 0.2])
weights1 = tf.reshape(weights1, [1,2])
biases1 = tf.reshape(biases1, [2])
allWeights.append(weights1)
allBiases.append(biases1)

weights2 = tf.Variable([[0.1, 0.3], [0.5, 0.2]])
biases2 = tf.Variable([0.3, 0.4])
weights2 = tf.reshape(weights2, [2,2])
biases2 = tf.reshape(biases2, [2])
allWeights.append(weights2)
allBiases.append(biases2)

weights3 = tf.Variable([0.8, 0.7])
biases3 = tf.Variable([0.6])
weights3 = tf.reshape(weights3, [2,1])
biases3 = tf.reshape(biases3, [1])
allWeights.append(weights3)
allBiases.append(biases3)

data = tf.Variable([0.9])
data = tf.reshape(data, [1,1])

preActivate1 = tf.add(tf.matmul(data, weights1), biases1)
activate1 = tf.nn.sigmoid(preActivate1)

preActivate2 = tf.add(tf.matmul(activate1, weights2), biases2)
activate2 = tf.nn.sigmoid(preActivate2)

preActivate3 = tf.add(tf.matmul(activate2, weights3), biases3)
activate3 = preActivate3


with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print sess.run(preActivate1)
    
    """with open('exampleNN2hl.dat', 'w') as outFile:
        outStr = "%1d %1d sigmoid" % (nLayers, nNodes)
        outFile.write(outStr + '\n')
        size = len(allWeights)
        for i in range(size):
            weights = sess.run(allWeights[i])
            if i < size-1:
                for j in range(len(weights)):
                    for k in range(len(weights[0])):
                        outFile.write("%.12g" % weights[j][k])
                        outFile.write(" ")
                    outFile.write("\n")
            else:
                for j in range(len(weights[0])):
                    for k in range(len(weights)):
                        outFile.write("%.12g" % weights[k][j])
                        outFile.write(" ")
                    outFile.write("\n")
                
        outFile.write("\n")
            
        for biasVariable in allBiases:
            biases = sess.run(biasVariable)
            for j in range(len(biases)):
                outFile.write("%.12g" % biases[j])
                outFile.write(" ")
            outFile.write("\n")"""
            
            
            
            
            