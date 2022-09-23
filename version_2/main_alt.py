import time
import pygame

import numpy as np

from network import empty_networks, propagate_parallel

from numba import njit, prange

@njit(cache=True,parallel=True)
def set_target(y, x=np.array([[]])):
    for i in prange(x.shape[0]):
        value = y[ int(x[i,7]) ]
        x[i,0:2] = value


@njit(cache=True, parallel=True)
def calculate_fitness(creatures:np.ndarray, output:np.ndarray):
    for i in prange(creatures.shape[0]):
        creature = creatures[i]
        frames, collected, alive, distance = creature[6:10]
        # Frametime, Collected, Alive, Closest distance
        mults = (1, 20, 100, 10)

        output[i] = frames*mults[0] + collected*mults[1] - alive*mults[2] + (mults[3]-distance)


def reset_creatures(creatures):
    creatures[:,:] = 0 # Reset all values to 0
    creatures[:,2:4] = middle # All starting positions in the middle of bounds
    creatures[:,8] = 1 # Set all creatures to be alive
    creatures[:,9] = 9999 # set all closest distance to infinite


def evolve_networks(weights:list, biases:list, fitnesses:np.ndarray, mutate_rate:float):
    pass

#                  x,    y
bounds = np.array([300,  150])
draw_scale = 5
middle = bounds//2

# Initialize pygame
pygame.init()

screen = pygame.display.set_mode(bounds*draw_scale)

# 100 networks of 4 inputs, 2 hidden layers, and 2 outputs
network_amount = 2000
biases, weights = empty_networks(network_amount, 4, [2,2,2], 2)

#    0        1        2     3     4        5        6       7          8      9
# TargetX, TargetY; PosX, PosY; MotionX, MotionY; Frames, Collected, Alive; Closest;
creatures = np.zeros((network_amount, 10))
reset_creatures(creatures)

creature_colors = np.random.randint(0, 255, (network_amount,3))

creature_speed = 3

collect_distance = 1
collect_time = 3

target_fps = 60


inputs = np.array([
    [100,75],
    [200,75],
    [75,100],
    [75,50]
])

max_collected = len(inputs)

total = time.time()

for frame in range(2000):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

    screen.fill((0,0,5))

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
    distance[:,:] /= bounds/10

    # Set closest distance
    sum_abs_distance = np.sum(abs_distance,1)
    mask = creatures[:,9] > sum_abs_distance
    creatures[:,9] *= (mask-1)*-1
    creatures[:,9] += sum_abs_distance * mask

    # Add frame time to creatures with distance less than max-distance
    creatures[:,6] += np.logical_and(abs_distance[:,0] < collect_distance, abs_distance[:,1] < collect_distance)

    # Add 1 to collected creatures with frame time > collect_time
    creatures[:,7] += creatures[:,6] >= collect_time

    # Reset frame time
    creatures[:,6] *= creatures[:,6] < collect_time

    # Get creatures' Motion
    motion = creatures[:,4:6]

    # Combine target offsets & thrust into one array
    ## [Ox, Oy, Mx, My]
    inps = np.append(distance[:], motion, 1)
    inps = np.reshape(inps, (network_amount, 1, 4))

    # Forward propagate using offset/motion inputs
    prepared_outputs = [inps]+[np.zeros(x.shape) for x in biases]
    new_motion = propagate_parallel(prepared_outputs, weights, biases)

    if np.isnan(np.sum(new_motion)):
        print("NANs detected")

    # Add output thrust to all creatures' motion
    new_motion = np.reshape(new_motion, (network_amount, 2)) * creature_speed
    creatures[:,4:6] += new_motion / target_fps
    creatures[:,4:6] *= alive

    # Check creatures within box
    within_box = np.sum(np.abs(np.floor(creatures[:, 2:4] / bounds)), 1)
    within_box = np.reshape(within_box, (network_amount,1))

    # Set creatures outside of bounding box or with max collected as dead
    alive = np.logical_and(within_box[:,0] == 0, creatures[:,7] <= max_collected)
    creatures[:,8] = alive

    # Update creature position
    # print(creatures[0,2:4])
    creatures[:,2:4] += creatures[:,4:6]
    targets = creatures[:,0:2]
    targets = np.unique(targets, axis=0)

    for i, position in enumerate(creatures[:,2:4]):
        position = position*draw_scale
        
        pygame.draw.circle(screen, creature_colors[i], position, draw_scale)

    for position in targets:
        position = position*draw_scale

        pygame.draw.circle(screen, (255,255,0), position, draw_scale/2)
    
    # Check if all creatures are dead
    if np.sum(creatures[:,8]) == 0:
        print("All creatures died")
        break

    
    pygame.time.delay(1)
    pygame.display.flip()
    end = time.time() - start
    print(f"{1/end:.2f}, {end:.4f}")

fitnesses = np.zeros(network_amount)

calculate_fitness(creatures, fitnesses)

total = time.time() - total
indexes = np.argsort(fitnesses)

for i in indexes[-100:]:
    print(fitnesses[i])

print(f"Total time: {total:.4f}s")
print(f"Total frames: {frame+1}")
print(f"Average time per frame: {total/(frame+1):.4f}")