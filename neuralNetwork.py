import tensorflow as tf
import numpy as np


class neuralNetwork:
    
    def __init__(self, nNodes, nLayers, activation, inputs=1, outputs=1,
                 weightsInit='trunc_normal', biasesInit='trunc_normal',
                 stdDev=0.1, constantValue=0.1):
        
        self.nNodes         = nNodes
        self.nLayers        = nLayers
        self.activation     = activation
        self.inputs         = inputs
        self.outputs        = outputs
        self.weightsInit    = weightsInit
        self.biasesInit     = biasesInit
        self.stdDev         = stdDev
        self.constantValue  = constantValue
        
        self.allWeights         = []
        self.allBiases          = []
        self.allPreActivations  = []
        self.allActivations     = []
        
    def init_weights(self, shape):
      
        if self.weightsInit == 'normal':
            return tf.Variable( tf.random_normal(shape, stddev=self.stdDev) )
            
        elif self.weightsInit == 'trunc_normal':
            return tf.Variable( tf.truncated_normal(shape, stddev=self.stdDev) )
            
        elif self.weightsInit == 'uniform':
            return tf.Variable( tf.random_uniform(shape, minval=-0.1, maxval=0.1))
            
        elif self.weightsInit == 'xavier':
            fanIn  = shape[0]
            fanOut = shape[1]
            low = -np.sqrt(6.0/(fanIn + fanOut)) # {sigmoid:4, tanh:1} 
            high = np.sqrt(6.0/(fanIn + fanOut))
            return tf.Variable( tf.random_uniform(shape, minval=low, maxval=high) )
            
        else:
            print '%s is not a valid weight initialization method' % self.weightsInit
            exit(1)

            
            
    def init_biases(self, shape):
    
        if self.biasesInit == 'zeros':
            return tf.Variable( tf.zeros(shape) )
            
        elif self.biasesInit == 'normal':
            return tf.Variable( tf.random_normal(shape, stddev=self.stdDev) )
            
        elif self.biasesInit == 'constant':
            return tf.Variable( tf.constant(self.constantValue, shape=shape) )
            
        elif self.biasesInit == 'trunc_normal':
            return tf.Variable( tf.truncated_normal(shape, stddev=self.stdDev) )
               
        else:
            print '%s is not a valid weight initialization method' % self.biasesInit
            exit(1)
        
    
    def variable_summaries(self, var, name):
        """Attach a lot of summaries to a Tensor."""
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean/' + name, mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('sttdev/' + name, stddev)
            tf.summary.scalar('max/' + name, tf.reduce_max(var))
            tf.summary.scalar('min/' + name, tf.reduce_min(var))
            tf.summary.histogram(name, var)           
            
    
    def nn_layer(self, input_tensor, input_dim, output_dim, layer_name, activation):
        """Reusable code for making a simple neural net layer.
        It does a matrix multiply, bias add, and then uses relu to nonlinearize.
        It also sets up name scoping so that the resultant graph is easy to read,
        and adds a number of summary ops.
        """
        
        # Adding a name scope ensures logical grouping of the layers in the graph.
        with tf.name_scope(layer_name):
        
            with tf.name_scope("weights"):
                weights = self.init_weights([input_dim, output_dim])
                self.allWeights.append(weights)
                self.variable_summaries(weights, layer_name + "/weights")
                
            with tf.name_scope("biases"):
                biases = self.init_biases([output_dim])
                self.allBiases.append(biases)
                self.variable_summaries(biases, layer_name + "/biases")
                
            with tf.name_scope("Wx_plus_b"):
                preactivate = tf.matmul(input_tensor, weights) + biases
                self.allPreActivations.append(preactivate)
                tf.summary.histogram(layer_name + "/pre_activations", preactivate)
                
            if not activation == None:
                activations = activation(preactivate, "activation")
                self.allActivations.append(activations)
                tf.summary.histogram(layer_name + "/activations", activations)
                return activations
            else:
                activations = tf.identity(preactivate, "activation")
                self.allActivations.append(activations)
                return activations
                
        
    def model(self, data): 
        
        nNodes  = self.nNodes
        nLayers = self.nLayers
        inputs  = self.inputs
        outputs = self.outputs
        
        activations = []
        
        hidden1 = self.nn_layer(data, inputs, nNodes, "layer1", self.activation)
        activations.append(hidden1)
        
        # following layers
        for layer in range(1, nLayers, 1):
            layerName = "layer%1d" % (layer+1)
            
            # make sure activation between last hidden layer an output is a sigmoid
            if layer == nLayers-1:
                if not (self.activation == tf.nn.sigmoid or self.activation == tf.nn.tanh) :
                    act = self.nn_layer(activations[layer-1], nNodes, nNodes, layerName, tf.nn.sigmoid)
                else:
                    act = self.nn_layer(activations[layer-1], nNodes, nNodes, layerName, self.activation)
            else:       
                act = self.nn_layer(activations[layer-1], nNodes, nNodes, layerName, self.activation)
            activations.append(act)
               
        outputLayer = self.nn_layer(activations[nLayers-1], nNodes, outputs, "outputLayer", None)
        activations.append(outputLayer)
        
        return outputLayer


    

    
    
    
    
    
    
    
    
    
