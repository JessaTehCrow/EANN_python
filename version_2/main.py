import time

import numpy as np
from network import empty_networks, propagate_parallel

from numba import njit, prange

@njit(cache=True,parallel=True)
def set_target(y, x=np.array([[]])):
    for i in prange(x.shape[0]):
        value = y[ int(x[i,7]) ]
        x[i,0:2] = value

# 100 networks of 4 inputs, 2 hidden layers, and 2 outputs
network_amount = 200
biases, weights = empty_networks(network_amount, 4, [3,3], 2)

# TargetX, TargetY. PosX, PosY. MotionX, MotionY. Frames, Collected, Alive.
creatures = np.zeros((network_amount, 9))
creatures[:,8] = 1 # Set all creatures to be alive
creature_speed = 3

collect_distance = 0.1
collect_time = 5

target_fps = 60

#                  x,    y
bounds = np.array([300,  150])
middle = bounds//2

creatures[:,2:4] = middle

inputs = np.array([
    [100,75],
    [2,2],
    [3,3],
    [4,4]
])

max_collected = len(inputs)

total = time.time()

for frame in range(1000):
    start = time.time()

    # Set new targets for creatures
    set_target(inputs, creatures)

    # Get creature alive states
    alive = np.reshape(creatures[:,8], (network_amount,1))

    # Get creature target offsets
    ## [Tx-Px, Ty-Py]
    distance = np.reshape(creatures[:,0:4], (network_amount, 2,2))
    distance = np.subtract.reduce(distance, 1)
    abs_distance = np.abs(distance)

    # Add frame time to creatures with distance less than max-distance
    creatures[:,6] += np.logical_and(abs_distance[:,0] < collect_distance, abs_distance[:,1] < collect_distance)

    # Add 1 to collected creatures with frame time > collect_time
    creatures[:,7] += creatures[:,6] >= collect_time

    # Reset frame time
    creatures[:,6] *= creatures[:,6] < collect_time

    # Get creatures' Motion
    motion = creatures[:,4:6] * alive

    # Combine target offsets & thrust into one array
    ## [Ox, Oy, Mx, My]
    inps = np.append(distance[:], motion, 1)
    inps = np.reshape(inps, (network_amount, 1, 4))

    # Forward propagate using offset/motion inputs
    prepared_outputs = [inps]+[np.zeros(x.shape) for x in biases]
    new_motion = propagate_parallel(prepared_outputs, weights, biases)

    # Add output thrust to all creatures' motion
    new_motion = np.reshape(new_motion, (network_amount, 2)) * creature_speed
    creatures[:,4:6] += new_motion / target_fps

    # Check creatures within box
    within_box = np.sum(np.floor(creatures[:, 2:4] / bounds))

    # Set creatures outside of bounding box or with max collected as dead
    dead = np.logical_or(within_box == 0, creatures[:,7] >= max_collected)
    creatures[:,8] = dead

    # Update creature position
    # print(creatures[0,2:4])
    creatures[:,2:4] += creatures[:,4:6]

    # Check if all creatures are dead
    if np.sum(creatures[:,8]) == 0:
        print("All creatures died")
        break

    end = time.time() - start
    print(f"{1/end:.2f}, {end:.4f}")

total = time.time() - total
print(f"total time: {total:.4f}s")
print(f"Average time per frame: {total/frame:.4f}")

print(creatures[:1000,7])