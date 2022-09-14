import random, os, json, time
import pygame

from util import root_directory
from learning_envs.dot.game import Game
from util.network import Network


AI_DIR = "/learning_envs/dot/generations/"

def main(argv):
    if argv[2:]:
        random_generation = int(argv[2])
    else:
        random_generation = random.randint(0, 10000)
    print('RNG used:', random_generation)

    # Game information
    game_width = 300
    game_height = 200
    creature_size = 1
    target_size = 0.3
    max_iterations = 999999

    GTDR = 5 # Game to display ratio
    gravity = 3.6
    game = Game(game_width, game_height, gravity, max_iterations)
    game.max_dots = 12

    # Pygame initialization
    pygame.init()
    screen = pygame.display.set_mode([game_width*GTDR, game_height*GTDR])
    font = pygame.font.SysFont("unispacebold", 10)

    # Load best_ais
    files = os.listdir(root_directory + AI_DIR)
    ai_files = [file for file in files if file.startswith("best_")]

    for ai_file in ai_files:
        with open(root_directory + AI_DIR + ai_file, 'r') as f:
            data = json.load(f)
        
        network = Network(len(data[1][0]), data[1], data[2])
        network._generation = data[0]

        game.register_network(network, 5, 1, random_generation)
    
    # Do thing
    while not game.next():
        # pygame event handler
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit("USER EXIT")
        
        screen.fill(0x111111)
        for ai in game.ais:
            pygame.draw.circle(screen, ai.color, [ai.pos[0]*GTDR, ai.pos[1]*GTDR], creature_size*GTDR)
            pygame.draw.circle(screen, ai.color, [ai.target[0]*GTDR, ai.target[1]*GTDR], target_size*GTDR)

            img = font.render(f"{ai.nn._generation}", True, (150, 150, 150))
            screen.blit(img, (ai.pos[0]*GTDR - (creature_size*GTDR*2), ai.pos[1]*GTDR - (creature_size*GTDR*4)))

            img = font.render(f"{ai.collected}", True, (0,200,0))
            screen.blit(img, (ai.pos[0]*GTDR - (creature_size*GTDR)-2, ai.pos[1]*GTDR + (creature_size*GTDR*2)))
        
        pygame.display.flip()
        if game.iteration == 1:
            time.sleep(2)

        time.sleep(1/120)
    
    time.sleep(2)