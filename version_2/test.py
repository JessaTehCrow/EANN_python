import numpy as np

from numba import njit, prange

def weird_function(inputs:np.ndarray, weights:np.ndarray, output:np.ndarray):
    shape = (inputs[0].shape[1], output[0].shape[1])

    for x in prange(inputs.shape[0]):
        input = inputs[x].T
        weight = weights[x]

        mult = np.sum(input * weight,1)

        output[x] = mult


# Setting up variables
inputs  = np.random.rand( 1000, 1, 4  )
weights = np.ones( (1000, 1, 12) )
output = np.zeros( (1000, 1, 4)  )

# Preparing both parallel and series JIT functions
error = njit(weird_function, parallel=True)
no_error = njit(weird_function, parallel=False)

# Running tests
print("Running non-jitted")
weird_function(inputs, weights, output)


print("Running Jiitted non-parallel")
no_error(inputs, weights, output)


print("Running Jitted parallel (With error)")
error(inputs, weights, output)