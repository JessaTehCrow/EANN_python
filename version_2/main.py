import time

import numpy as np

import network, activation

# 100 networks of 4 inputs, 2 hidden layers, and 2 outputs
network_amount = 1000
biases, weights = network.empty_networks(network_amount, 4, [3,3], 2)

# Print shapes

outputs = [[np.random.rand(1,4)]] + [np.zeros(x.shape) for x in biases]


weight = weights[0][0]

activation_function = activation.gaussian

activation_function(3)

start = time.time()

for x in range(network_amount):
    for i,_ in enumerate(weights):

        weight = weights[i][x]
        bias = biases[i][x]
        inputs = outputs[i][0 if not i else x]

        out = network.get_output(inputs, weight, bias, activation_function)

        outputs[i+1][x] = activation_function(out)

end = time.time() - start

print(f"total single prediction for 1000 NNs execution time: {end:.4f}s")