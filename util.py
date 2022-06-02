import math

# https://en.wikipedia.org/wiki/Sigmoid_function
def sigmoid(x):
  return 1 / (1 + math.exp(-x))