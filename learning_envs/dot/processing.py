import json, os, time

from util import root_directory
from learning_envs.dot.game import Game
from util.network import random_network, evolve, Network

from typing import List
from multiprocessing import Process, Manager
from threading import Thread

def process_function(game_options:list, ai_options:list, ais:list, outputs:list, process:int):
    game = Game(*game_options)

    game.load_network(ais, *ai_options)

    while not game.next():
        continue

    outputs[process] = game.ais


def main(argv):
    # Game information
    game_width = 300
    game_height = 200
    max_iterations = 6000
    gravity = 3.6
    game_options = [game_width, game_height, gravity, max_iterations]

    # Define NN values
    inputs        = 4
    hidden        = [10, 10]
    hidden_layers = len(hidden)
    outputs       = 4
    mutate_rate   = 0.05

    # Define evolve information
    export_folder = root_directory+"/learning_envs/dot/generations/"
    creatures = 200
    generations = 2000000
    gen_start = 0

    # Define processing settings
    process_blocks = 10

    # Generate first generation
    if len(argv[1:])==3 and argv[2] == "load":
        print(f"Loading generation {argv[3]}")
        gen_to_load = int(argv[3])
        filename = f"generation_{gen_to_load}.json"
        if not os.path.isfile(export_folder+filename):
            exit(f"Unable to load generation {gen_to_load}. File does not exist")
        gen_start = gen_to_load
        
        with open(export_folder+filename, 'r') as f:
            data = json.load(f)
        
        ais = []
        for generation, fitness, (weights,biases) in data:
            ais.append( Network(inputs, weights, biases))
    else:
        ais = [random_network(inputs, hidden_layers, hidden, outputs) for ai in range(creatures)]
    
    # Generation thingy
    outputs = {}
    # manager = Manager()
    exec_start = time.time()
    for generation in range(generations):
        gen_time = time.time()
        print(f"{'='*20} GENERATION {generation+gen_start} {'='*20}")
        ai_options = [1, 5, generation+gen_start]
        processes:List[Process] = []
        ais_per_block = len(ais)//process_blocks
        
        # Prepare ais
        # outputs = manager.dict()

        for block in range(8):
            process_ais = ais[block*ais_per_block:(block+1)*ais_per_block]
            p = Process(target=process_function, args=(game_options, ai_options, process_ais, outputs, block))
            processes.append(p)
        
        for process in processes:
            process.start()

        # Wait for ais to be done
        for process in processes: 
            process.join()
        
        # Handle processes output's
        fin = Game(*game_options)
        for ai_list in outputs.values():
            fin.ais += ai_list
    
        top_performers = fin.get_top_ais(creatures//4)

        # Save ais / generation
        if not (generation+gen_start)%10:
            avg_fitness = sum([ai.fitness for ai in fin.ais])/len(top_performers)
            networks = [[generation,avg_fitness,[ai.nn.weights, ai.nn.biases]] for ai in fin.ais]

            with open(f"{export_folder}generation_{generation+gen_start}.json", 'w') as f:
                json.dump(networks,f)
        
        if not (generation+gen_start)%100:
            best = top_performers[0]
            with open(export_folder+f"best_{generation+gen_start}.json", 'w') as f:
                json.dump([generation, best.nn.weights, best.nn.biases], f)

        ais = evolve(top_performers, creatures, mutate_rate, True)

        print(f"Best performance:",top_performers[0].fitness)
        print(f"Avg top fitness :", sum([ai.fitness for ai in top_performers])/len(top_performers))
        print(f"Execution time  :", time.time() - gen_time)
        print(f"Total exec time :", time.time() - exec_start)
        print()