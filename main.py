from network import random_network, mate, Network
"""
This file is mainly used for personal debugging whilst creating this repo.
This file might get removed or used for other purposes later down the line.
"""

# Neural network inputs
inputs        = 1
hidden_layers = 1
hidden        = 2
outputs       = 1

# Parent neural networks
network1 = Network(1,[[[1,1]],[[1],[1]],[None]], [[None],[1,1], [1]])
network2 = Network(2,[[[2,2]],[[2],[2]],[None]], [[None],[2,2], [2]])


# Debug mating
child = mate(network1,network2, 0.03)

# Get new values
print(child.biases)
print(child.weights)