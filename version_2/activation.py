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
    exp = np.exp(x)
    neg_exp = np.exp(-x)
    return (exp-neg_exp) / (exp+neg_exp)


@vectorize
def SELU(x):
    l = 1.0507 
    a = 1.67326

    return l * (a*(np.exp(x)-1) if x<=0 else x)

@vectorize
def RELU(x):
    return 0 if x<0 else x
