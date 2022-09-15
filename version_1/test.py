import math, time

import numpy as np

from sys import getsizeof
from numba import jit, vectorize, typed
from numba import cuda


def consitent_shape(array:list):
    # Check if already numpy / typed, then just use that
    if isinstance(array, (np.ndarray, typed.List)):
        return True

    length = len(array)
    if [*map(type, array)].count(type(array[0])) != length:
        try:
            array = [*map(float,array)]
        except Exception:
            return False
    
    elif type(array[0]) in (float, int):
        return [length]
    
    elif type(array[0]) != list:
        return False
    
    prev_shape = consitent_shape(array[0])
    for x in array[1:]:
        if consitent_shape(x) != prev_shape:
            return False

    return True


def to_typed(array:list):
    new_array = []
    for arr in array:
        if consitent_shape(arr):
            new_array.append(np.array(arr))
        else:
            new_array.append(to_typed(arr))

    return typed.List(new_array)


def type_map(f):
    def inner(*args,**kwargs):
        new_args = []
        for arg in args:
            if isinstance(arg, list):
                arg = np.array(arg) if consitent_shape(arg) else to_typed(arg)
            new_args.append(arg)

        return f(*new_args, **kwargs)
    return inner


def get_network(layers:list):
    largest = max(layers[1:])
    weights = np.zeros([len(layers)-1, largest, largest])
    biases = np.zeros([len(layers)-1, largest])

    for index, (depth, neurons) in enumerate(zip(layers[1:], layers[:-1])):
        weight = np.random.uniform(-1,1, [largest, largest])
        bias = np.random.uniform(-1,1, [largest])

        weight[:, neurons:] = 0
        weight[depth:, :] = 0
        bias[depth:] = 0

        weights[index] = weight
        biases[index] = bias

    return weights,biases


@cuda.jit(device=True)
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

@cuda.jit(device=True)
def sigmoid_matmult(A:np.ndarray, B:np.ndarray):
    output = cuda.device_array( B.shape[1] )
    
    for x, v1 in enumerate(A):
        for y, v2 in enumerate(B[x]):
            output[y] += v1*v2

    return output

@cuda.jit(device=True)
def forward(inputs, weights, biases):
    prev = inputs

    for weight in weights:
        prev = sigmoid_matmult(prev, weight)
        
    return prev


@cuda.jit
def forward_cuda(inputs:np.ndarray, weights:np.ndarray, biases:np.ndarray, outputs:np.ndarray):
    index = cuda.grid(1)

    # Check if within bounds
    if index > inputs.shape[0]:
        return

    input  = inputs[index]
    weight = weights[index]
    bias   = biases[index]

    output_shape = outputs.shape[0]
    output = forward(input, weight, bias)
    outputs[index] = output[:output_shape]


layers = [2,3,3,2]
network_am = 10_000

networks = [get_network(layers) for _ in range(network_am)]
inputs = [np.random.uniform(-1,1,[layers[0]]) for _ in range(network_am)]

weight_shape = networks[0][0].shape
bias_shape   = networks[0][1].shape

inp_weights = np.zeros([network_am, *weight_shape])
inp_biases  = np.zeros([network_am, *bias_shape])
inp_inputs  = np.zeros([network_am, layers[0]])
output = np.zeros([network_am, layers[-1]])

for index, ((weights, biases), input) in enumerate(zip(networks, inputs)):
    inp_weights[index] = weights
    inp_biases[index]  = biases
    inp_inputs[index] = input



start = time.time()
threads = 1028
blocks  = math.ceil(network_am / threads)

forward_cuda[blocks, threads](inp_inputs, inp_weights, inp_biases, output)

end = time.time() - start
print(f"{end:.3f} seconds")