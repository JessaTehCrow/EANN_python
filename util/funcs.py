import math

from numba import jit

# https://en.wikipedia.org/wiki/Sigmoid_function
@jit
def sigmoid(x):
  return 1 / (1 + math.exp(-x))