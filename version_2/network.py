import numpy as np
import activation

from typing import List
from numba import njit, prange


def empty_networks(amount:int, inputs:int, hidden_layers:list, outputs:int) -> List[List[np.ndarray]]:
    range = 5
    biases  = [(np.random.rand(amount,1,x)-0.5)*2*range for x in [*hidden_layers, outputs]]
    weights = [(np.random.rand(amount,1,x*y)-0.5)*2*range for x,y in zip([inputs]+hidden_layers, hidden_layers+[outputs])]

    return tuple(biases), tuple(weights)


def propagate_parallel(outputs:tuple, weights:tuple, biases:tuple):
    weight_len = len(weights)

    for i,(w,b) in enumerate(zip(weights,biases)):
        inputs = outputs[i]

        if i==weight_len-1:
            matmult_tanH_parallel(inputs, w, b, outputs[i+1])
        else:
            matmult_ReLU_parallel(inputs, w, b, outputs[i+1])

    return outputs[-1]


def propagate(input:np.ndarray, weights:list, biases:list):
    outputs = [input]+[np.zeros(x.shape) for x in biases]
    weight_len = len(weights)

    for x in range(biases[0].shape[0]):
        for i,(w,b) in enumerate(zip(weights,biases)):

            weight = w[x]
            bias = b[x]
            inputs = outputs[i][x]

            if i==weight_len-1:
                out = matmult_tanH(inputs, weight, bias)
            else:
                out = matmult_ReLU(inputs, weight, bias)

            outputs[i+1][x] = out

    return outputs[-1]


@njit(cache=True)
def matmult_ReLU(inputs:np.ndarray, weights:np.ndarray, biases:np.ndarray):
    out = np.reshape(np.sum(inputs.T*weights, 0), (inputs.shape[1], biases.shape[1]))
    out2 = np.reshape(np.sum(out, 0), biases.shape)

    return activation.RELU(out2 + biases)


@njit(cache=True)
def matmult_tanH(inputs:np.ndarray, weights:np.ndarray, biases:np.ndarray):
    out = np.reshape(np.sum(inputs.T*weights, 0), (inputs.shape[1], biases.shape[1]))
    out2 = np.reshape(np.sum(out, 0), biases.shape)

    return activation.tanH(out2 + biases)




@njit(cache=True, parallel=True)
def matmult_ReLU_parallel(inputs:np.ndarray, weights:np.ndarray, biases:np.ndarray, output:np.ndarray):
    shape = (inputs[0].shape[1], biases[0].shape[1])

    for x in prange(inputs.shape[0]):
        input = inputs[x].T
        weight = weights[x]
        bias = biases[x]

        out = np.reshape(np.sum(input @ weight, 0), shape)
        out2 = np.reshape(np.sum(out, 0), bias.shape)

        output[x] = activation.RELU(out2 + bias)


@njit(cache=True, parallel=True)
def matmult_tanH_parallel(inputs:np.ndarray, weights:np.ndarray, biases:np.ndarray, output:np.ndarray):
    shape = (inputs[0].shape[1], biases[0].shape[1])

    for x in prange(inputs.shape[0]):
        input = inputs[x].T
        weight = weights[x]
        bias = biases[x]

        out = np.reshape(np.sum(input @ weight, 0), shape)
        out2 = np.reshape(np.sum(out, 0), bias.shape)

        output[x] = activation.tanH(out2 + bias)