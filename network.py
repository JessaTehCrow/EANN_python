import random
import numpy as np

from copy import deepcopy
from util import sigmoid
from typing import List

class Network():
    def __init__(self, inputs:int, weights:list, biases:list):
        self._debug = False
        self.inputs = inputs
        self.weights = weights
        self.biases  = biases
    

    # Debugger
    def log(self, *text) -> None:
        if self._debug:
            print(*text)


    # Predict the solution to the input problem/senario
    def predict(self, inputs:list) -> list:
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
        out = [False]*len(prev_layer)
        heighest = max(prev_layer)
        out[prev_layer.index(heighest)] = True
        # avg = sum(prev_layer)/len(prev_layer)
        # out = [x>(avg*1.1) for x in prev_layer]
        return out


# Prepare a neural network with random inital weights/biases.
def random_network(inputs:int, hidden_layers:int, hidden:int, outputs:int) -> Network:
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
def mate(parent1:Network, parent2:Network, mutate_rate:float) -> Network:
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


# Evolve an entire group of networks
def evolve(networks:list, new_length:int, mutate_rate:float, elitism:bool=False) -> List[Network]:
    def random_parent(other_parent:Network=None) -> Network:
        best = other_parent

        while best == other_parent:
            ais = [random.choice(networks) for _ in range(10)]
            best = max(ais, key=lambda ai: ai.fitness)
            # print(best.nn)
            best = best.nn

        return best
    
    new_ais = []
    for ai in range(new_length):
        parent1 = random_parent()
        parent2 = random_parent(parent1)

        ai = mate(parent1, parent2, mutate_rate)
        new_ais.append(ai)
    
    if elitism:
        new_ais[-1] = networks[0].nn

    return new_ais