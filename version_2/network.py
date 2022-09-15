import numpy as np

from typing import List
from numba import jit


def empty_networks(amount:int, inputs:int, hidden_layers:list, outputs:int) -> List[List[np.ndarray]]:
    biases = [np.random.rand(amount,1,x) for x in [*hidden_layers, outputs]]
    weights = [np.random.rand(amount,1,x*y) for x,y in zip([inputs]+hidden_layers, hidden_layers+[outputs])]

    return biases, weights


def get_output(inputs:np.ndarray, weights:list, biases:list, activation_function:callable):
    out = np.reshape(sum(inputs.T*weights), [len(inputs[0]),len(biases[0])])
    out2 = np.reshape(sum(out), [1,len(biases[0])])

    return activation_function(out2 + biases)