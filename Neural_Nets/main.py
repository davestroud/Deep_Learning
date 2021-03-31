
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from keras.datasets import mnist





# Initialize a network object
class Network(object):
    
    # The bias and weights are all initialized randomly
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes # contains the number of neurons in a layer
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        
# Create a network with 2 neurons in the first layer
# Three neurons in the second layer
# and one neuron in the final layer
net = Network([2, 3, 1])

# Converts a real value in to one that can be interpreted 
# as a probability
def sigmoid(z):
    return 1.0/(1.0+ np.exp(-z))

def feedforward(self, a):
    """Return the output of the network if "a" is input."""
    for b, w in zip(self.biases, self.weights):
        a = sigmoid(np.dot(w, a) + b) # a' = sigmoid(wa + b)
    return a
        
def SGD(self, training):
    pass
        
        