from network import random_network, mate, Network
from game import Game
"""
This file is mainly used for personal debugging whilst creating this repo.
This file might get removed or used for other purposes later down the line.
"""

# Neural network inputs
inputs        = 1
hidden_layers = 1
hidden        = 2
outputs       = 2

# Parent neural networks
network1 = random_network(inputs, hidden_layers, hidden, outputs)

game = Game(150, 100, 3.6)
game.register_network(network1, 5)

# Get new values
game.next()