
from keras.datasets import mnist
import numpy as np


# Initialize a network object
class Network(object):
    
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
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
        
        
        