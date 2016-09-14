"""
Make neural network model
"""

import tensorflow as tf

def model_1HiddenLayerRelu(data, nodesPerLayer, noInputs, noOutputs):
    
    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([noInputs, nodesPerLayer[0]])),
                      'biases':  tf.Variable(tf.random_normal([nodesPerLayer[0]]))}

    output_layer   = {'weights': tf.Variable(tf.random_normal([nodesPerLayer[0], noOutputs])),
                      'biases':  tf.Variable(tf.random_normal([noOutputs])),}
    
    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    output = tf.matmul(l1, output_layer['weights']) + output_layer['biases']  

    return output
    
    
def model_2HiddenLayersRelu(data, nodesPerLayer, noInputs, noOutputs):
    
    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([noInputs, nodesPerLayer[0]])),
                      'biases':  tf.Variable(tf.random_normal([nodesPerLayer[0]]))}

    hidden_2_layer = {'weights': tf.Variable(tf.random_normal([nodesPerLayer[0], nodesPerLayer[1]])),
                      'biases':  tf.Variable(tf.random_normal([nodesPerLayer[1]]))}

    output_layer   = {'weights': tf.Variable(tf.random_normal([nodesPerLayer[1], noOutputs])),
                      'biases':  tf.Variable(tf.random_normal([noOutputs])),}
    
    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    output = tf.matmul(l2, output_layer['weights']) + output_layer['biases']  

    return output    
    

def model_3HiddenLayersRelu(data, nodesPerLayer, noInputs, noOutputs):
    
    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([noInputs, nodesPerLayer[0]])),
                      'biases':  tf.Variable(tf.random_normal([nodesPerLayer[0]]))}

    hidden_2_layer = {'weights': tf.Variable(tf.random_normal([nodesPerLayer[0], nodesPerLayer[1]])),
                      'biases':  tf.Variable(tf.random_normal([nodesPerLayer[1]]))}

    hidden_3_layer = {'weights': tf.Variable(tf.random_normal([nodesPerLayer[1], nodesPerLayer[2]])),
                      'biases':  tf.Variable(tf.random_normal([nodesPerLayer[2]]))}

    output_layer   = {'weights': tf.Variable(tf.random_normal([nodesPerLayer[2], noOutputs])),
                      'biases':  tf.Variable(tf.random_normal([noOutputs])),}
    
    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']  

    return output
    
    
def model_1HiddenLayerSigmoid(data, nodesPerLayer, noInputs, noOutputs):
    
    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([noInputs, nodesPerLayer[0]])),
                      'biases':  tf.Variable(tf.random_normal([nodesPerLayer[0]]))}

    output_layer   = {'weights': tf.Variable(tf.random_normal([nodesPerLayer[0], noOutputs])),
                      'biases':  tf.Variable(tf.random_normal([noOutputs])),}
    
    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.sigmoid(l1)

    output = tf.matmul(l1, output_layer['weights']) + output_layer['biases']  

    return output


def model_2HiddenLayersSigmoid(data, nodesPerLayer, noInputs, noOutputs):
    
    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([noInputs, nodesPerLayer[0]])),
                      'biases':  tf.Variable(tf.random_normal([nodesPerLayer[0]]))}

    hidden_2_layer = {'weights': tf.Variable(tf.random_normal([nodesPerLayer[0], nodesPerLayer[1]])),
                      'biases':  tf.Variable(tf.random_normal([nodesPerLayer[1]]))}

    output_layer   = {'weights': tf.Variable(tf.random_normal([nodesPerLayer[1], noOutputs])),
                      'biases':  tf.Variable(tf.random_normal([noOutputs])),}
    
    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.sigmoid(l1)

    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.sigmoid(l2)

    output = tf.matmul(l2, output_layer['weights']) + output_layer['biases']  

    return output
    
    
def model_3HiddenLayersSigmoid(data, nodesPerLayer, noInputs, noOutputs):
    
    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([noInputs, nodesPerLayer[0]])),
                      'biases':  tf.Variable(tf.random_normal([nodesPerLayer[0]]))}

    hidden_2_layer = {'weights': tf.Variable(tf.random_normal([nodesPerLayer[0], nodesPerLayer[1]])),
                      'biases':  tf.Variable(tf.random_normal([nodesPerLayer[1]]))}

    hidden_3_layer = {'weights': tf.Variable(tf.random_normal([nodesPerLayer[1], nodesPerLayer[2]])),
                      'biases':  tf.Variable(tf.random_normal([nodesPerLayer[2]]))}

    output_layer   = {'weights': tf.Variable(tf.random_normal([nodesPerLayer[2], noOutputs])),
                      'biases':  tf.Variable(tf.random_normal([noOutputs])),}
    
    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.sigmoid(l1)

    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.sigmoid(l2)

    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.sigmoid(l3)

    output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']  

    return output
    