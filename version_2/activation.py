import numpy as np

from numba import jit

@jit
def sigmoid(x):
    return 1 / (1-np.exp(-x))


@jit
def bin_step(x):
    return x>0


@jit
def gaussian(x):
    return np.exp(-x**2)


@jit
def SiLU(x):
    return x/(1+np.exp(-x))


@jit
def tanH(x):
    exp = np.exp(x)
    neg_exp = np.exp(-x)
    return (exp-neg_exp) / (exp+neg_exp)


@jit
def SELU(x):
    l = 1.0507 
    a = 1.67326

    return l * (a*(np.exp(x)-1) if x<=0 else x)

@jit
def RELU(x):
    return 0 if x<0 else x