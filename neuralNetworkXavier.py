import tensorflow as tf
import numpy as np

np.random.seed(1)

def init_weights(shape, layer, init_method='normal', xavier_params = (None, None)):
    
    if init_method == 'zeros':
        return tf.Variable(tf.zeros(shape))
        
    elif init_method == 'normal':
        return tf.Variable(tf.random_normal(shape), name='b%1d' % layer)
        
    else: #xavier
        (fan_in, fan_out) = xavier_params
        low = -4*np.sqrt(6.0/(fan_in + fan_out)) # {sigmoid:4, tanh:1} 
        high = 4*np.sqrt(6.0/(fan_in + fan_out))
        return tf.Variable(tf.random_uniform(shape, minval=low, maxval=high), \
                           name='w%1d' % layer)
        

def model(data, num_hidden=10, num_layers=3, inputs=1, outputs=1): 
    
    weights = []    
    biases  = []
    neurons = []
    
    # first hidden layer
    w1 = init_weights([inputs, num_hidden], 1, 'xavier', xavier_params=(inputs, num_hidden))
    b1 = init_weights([num_hidden], 1, 'normal')
    h1 = tf.nn.sigmoid(tf.matmul(data, w1) + b1)
    weights.append(w1)
    biases.append(b1)
    neurons.append(h1)
    
    # following layers
    for layer in range(1, num_layers, 1):
        w = init_weights([num_hidden, num_hidden], layer+1, 'xavier', \
                       xavier_params=(num_hidden, num_hidden))
        b = init_weights([num_hidden], layer+1, 'normal')
        h = tf.nn.sigmoid(tf.matmul(neurons[layer-1], w) + b)
        weights.append(w)
        biases.append(b)
        neurons.append(h)
    
    # output layer
    w_o = init_weights([num_hidden, outputs], num_layers+1, 'xavier', \
                       xavier_params=(num_hidden, outputs))
    b_o = init_weights([outputs], num_layers+1, 'normal')
    h_o = tf.matmul(neurons[num_layers-1], w_o) + b_o
    weights.append(w_o)
    biases.append(b_o)    
    neurons.append(h_o)
    
    return h_o, weights, biases, neurons
    
    
if __name__ == '__main__':
    N = 100
    #data = np.linspace(0.9, 1.6, N, dtype='float32')
    data = np.random.uniform(0.9, 1.6, N)
    data.reshape([N,1])
    data.astype('float32')
    print data
    model(data)
    
    
    
    