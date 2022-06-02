from network import random_network, mate, Network

# Neural network inputs
inputs        = 1
hidden_layers = 1
hidden        = 2
outputs       = 1

# Get first neural network
network1 = Network(1,[[[1,1]],[[1],[1]],[None]], [[None],[1,1], [1]])
# network1._debug = True

network2 = Network(2,[[[2,2]],[[2],[2]],[None]], [[None],[2,2], [2]])


# Debugging
child = mate(network1,network2, 0.03)
print(child.biases)
print(child.weights)