import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

import pygame, json

from util import root_directory
from learning_envs.dot.game import Game

from util.network import random_network, evolve, Network

def main(argv):
    # Game information
    game_width = 300
    game_height = 200
    creature_size = 1
    target_size = 0.3
    max_iterations = 6000
    GTDR = 5 # Game to display ratio
    gravity = 3.6
    game = Game(game_width, game_height, gravity, max_iterations)

    # Initialize pygame
    pygame.init()
    screen = pygame.display.set_mode([game_width*GTDR, game_height*GTDR])
    font = pygame.font.SysFont("unispacebold", 25)
    best_font = pygame.font.SysFont("unispacebold", 15)
    RENDER = True

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

    # Generate first generation
    if len(argv[1:])==2 and argv[1] == "load":
        print(f"Loading generation {argv[2]}")
        gen_to_load = int(argv[2])
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
    averages = []

    import time
    start = time.time()
    for generation in range(gen_start, generations):
        game.reset()
        game.load_network(ais, 1, 5, generation)
        gen_length = len(str(generation))-1
        gen_modulo = 10**(len(str(gen_length)))
        save_generation = (generation%gen_modulo)==0
        best_generation = generation%100==0

        while not game.next():
            # pygame event handler
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    exit("USER EXIT")
                if event.type == pygame.KEYDOWN:
                    key = event.key
                    if key == pygame.K_r:
                        RENDER = not RENDER
                    elif key == pygame.K_s:
                        save_generation = True
                    elif key == pygame.K_b:
                        print("best gen enabled")
                        best_generation = True

            # Clear screen
            screen.fill(0x111111)

            def draw_text():
                collected = [creature.collected for creature in game.ais]
                text_draw = [f"Generation: {generation}",
                             f"Iteration: {game.iteration}",
                             f"Best Collected: {max(collected)}",
                             f"Total Collected: {sum(collected)}",
                             f"Dead: {game.dead}/{creatures}"]
                for i,text in enumerate(text_draw):
                    img = font.render(text, True, (0, 200, 255))
                    screen.blit(img,(10,10+(30*i)))
                
            if not RENDER:
                draw_text()
                pygame.display.flip()
                continue

            # Draw ais
            best_fitness = max([ai.fitness for ai in game.ais])
            best_drawn = False
            for ai in game.ais:
                pygame.draw.circle(screen, ai.color, [ai.pos[0]*GTDR, ai.pos[1]*GTDR], creature_size*GTDR)
                pygame.draw.circle(screen, ai.color, [ai.target[0]*GTDR, ai.target[1]*GTDR], target_size*GTDR)

                if ai.fitness == best_fitness and not best_drawn:
                    img = best_font.render("#1", True, (255,255,0))
                    screen.blit(img, (ai.pos[0]*GTDR, ai.pos[1]*GTDR-creature_size/2))
                    best_drawn = True

            draw_text()
            # Display to user
            pygame.display.flip()

        top_performers = game.get_top_ais(creatures//2)

        if save_generation:
            avg_fitness = sum([ai.fitness for ai in game.ais])/len(top_performers)
            networks = [[generation,avg_fitness,[ai.nn.weights, ai.nn.biases]] for ai in game.ais]

            with open(f"{export_folder}generation_{generation}.json", 'w') as f:
                json.dump(networks,f)
        
        if best_generation:
            best = top_performers[0]
            with open(export_folder+f"best_{generation}.json", 'w') as f:
                json.dump([generation, best.nn.weights, best.nn.biases], f)

        ais = evolve(top_performers, creatures, mutate_rate, True)


    # Quit pygame
    pygame.quit()