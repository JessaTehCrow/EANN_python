import random
import numpy as np

from copy import deepcopy
from util import sigmoid

class Network():
    def __init__(self, inputs:int, weights:list, biases:list):
        self._debug = False
        self.inputs = inputs
        self.weights = weights
        self.biases  = biases
    

    # Debugger
    def log(self, *text):
        if self._debug:
            print(*text)


    # Predict the solution to the input problem/senario
    def predict(self, inputs:list):
        if len(inputs) != self.inputs:
            exit(f"\n[ERROR] FAILED TO PREDICT NEURAL NETWORK: Inputs length incorrect. Expected {self.inputs}, got {len(inputs)}\n")
        
        self.log(f"Input length matches expected length ({self.inputs})")
        
        prev_layer = inputs
        for layer in range(len(self.biases[:-1])):
            temp_layer = []
            self.log(f"Layer {layer+1}/{len(self.biases)-1}: Process started")

            # Calculate values for each neuron
            for neuron in range(len(self.weights[layer+1])):
                value = 0
                bias = self.biases[layer+1][neuron]
                for w, weights in enumerate(self.weights[layer]):
                    value = prev_layer[w]*weights[neuron]

                value = sigmoid(value + bias)
                temp_layer.append(value)

                self.log(f"Neuron [{neuron}] bias: {bias}")
                self.log(f"Neuron [{neuron}]     = {value}")

            # Reset previous layer to current layer
            prev_layer = temp_layer
            self.log(f"Layer {layer+1}: Finished successfully\n")

        self.log(f"Prediction finished without errors")
        return prev_layer


# Prepare a neural network with random inital weights/biases.
def random_network(inputs:int, hidden_layers:int, hidden:int, outputs:int):
    # Prepare variables
    weights = [[0]*inputs, *[[0]*hidden for x in range(hidden_layers)], [None]*outputs]
    biases  = [[None]*inputs, *[[0]*hidden for x in range(hidden_layers)], [0]*outputs]

    # Populate weights
    for l, layer in enumerate(weights[:-1]):
        for n, neuron in enumerate(layer):
            temp_weights = []
            
            for weight in range(len(weights[l+1])):
                temp_weights.append(random.uniform(0, 1))
            
            weights[l][n] = temp_weights

    # Populate biases
    for l,layer in enumerate(biases[1:]):
        for n in range(len(layer)):
            layer[n] = random.uniform(-1, 1)

    # Return network
    return Network(inputs, weights, biases)



# Mate 2 (parent) Neural networks together
# This passes some genes of both parents (weights / biases) to the child, and mutates them slightly.
# The mutation allows for better solutions.
def mate(parent1:Network, parent2:Network, mutate_rate:float):
    parents = [parent1, parent2]
    weights = [None]*len(parent1.weights)
    biases  = [None]*len(parent1.weights)

    # Pass on genes
    for layer in range(len(parent1.weights)):
        parent = random.uniform(0,1) > 0.5
        weights[layer] = deepcopy(parents[parent].weights[layer])
        biases[layer]  = deepcopy(parents[parent].biases[layer])
    
    # Mutate genes
    #  Weights
    for l,layer in enumerate(weights[:-1]):
        for n,neuron in enumerate(layer):
            for w,weight in enumerate(neuron):
                weights[l][n][w] += np.random.normal(0, mutate_rate)

    #  Biases
    for l,layer in enumerate(biases[1:]):
        for n,neuron in enumerate(layer):
            biases[l+1][n] += np.random.normal(0, mutate_rate)
    
    return Network(parent1.inputs, weights, biases)