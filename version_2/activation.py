import numpy as np

from numba import njit, vectorize, prange

@vectorize
def sigmoid(x):
    return 1 / (1-np.exp(-x))


@vectorize
def bin_step(x):
    return x>0

@vectorize
def gaussian(x):
    return np.exp(-x**2)


@vectorize
def SiLU(x):
    return x/(1+np.exp(-x))


@vectorize
def tanH(x):
    return 1/(1+np.exp(-(2*x)))*2-1


@vectorize
def SELU(x):
    l = 1.0507 
    a = 1.67326

    return l * (a*(np.exp(x)-1) if x<=0 else x)

@vectorize
def RELU(x):
    return 0 if x<0 else x
