import numpy as np
import pygame

from network import Network, random_network
from typing import List


class Creature():
    def __init__(self, network:Network, thrust_power:float, map_size:list, gravity:float):
        self.nn = network

        self.size = map_size
        self.pos = [self.size[0]/2, self.size[1]/2]

        self.thrust = thrust_power
        self.gravity = gravity
        self.velocity = [0, 0]

        self.survived_steps = 0
        self.dead = False

    
    # Get the fitness of the AI
    def fitness(self):
        return self.survived_steps**1.1//1
    

    # Predict solution from senario input
    def predict(self, inputs:list):
        return self.nn.predict(inputs)
    

    # Next step
    def next(self):
        middle_offset = (self.size[1]/2 - self.pos[1])*2/self.size[1]

        predictions = self.predict([middle_offset])
        thrust = predictions[0] > predictions[1]
        forces = [0,thrust*self.thrust - self.gravity]

        self.velocity = [(x/60)+y for x,y in zip(forces,self.velocity)]
        self.pos = [x+y for x,y in zip(self.pos, self.velocity)]
        
        if not (0<self.pos[0]<self.size[0] and 0<self.pos[1]<self.size[1]):
            self.dead = True


class Game():
    def __init__(self, width:int, height:int, gravity:float):
        self.width = width
        self.height = height
        self.gravity = gravity
        self.ais:List[Creature] = []
    
    
    # Add network to game data
    def register_network(self, ai:Network, thrust_power:float):
        self.ais.append( Creature(ai, thrust_power, [self.width,self.height], self.gravity) )
    

    # Next step for all registered AIs
    def next(self):
        for ai in self.ais:
            ai.next()

        return all([ai.dead for ai in self.ais])
    

    # Remove all saved networks
    def reset(self):
        self.ais = []
    

    # Get top ais based on their fitness
    def get_top_ais(self, cut_off:int):
        top = sorted(self.ais, key=lambda ai: ai.fitness, reverse=True)[:cut_off]
        return top


if __name__ == "__main__":
    # Game information
    game_width = 150
    game_height = 100
    creature_size = 1
    GTDR = 7 # Game to display ratio
    gravity = 3.6
    game = Game(game_width, game_height, gravity)

    # Initialize pygame
    pygame.init()
    screen = pygame.display.set_mode([game_width*GTDR, game_height*GTDR])

    # Define NN values
    inputs        = 1
    hidden_layers = 1
    hidden        = 2
    outputs       = 1
    mutate_rate   = 0.03

    # Define evolve information
    creatures = 200
    generations = 2000

    # Generate first generation
    ais = [random_network(inputs, hidden_layers, hidden, outputs) for ai in range(creatures)]
    map(game.register_network, ais)

    for generation in generations:
        while not game.next():
            # pygame event handler
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    exit("USER EXIT")

            # Clear screen
            screen.fill(0x333333)

            # Draw ais
            for ai in game.ais:
                pygame.draw.circle(screen, 0xf4c7ff, [ai.pos[0]*GTDR, ai.pos[1]*GTDR], creature_size*GTDR)

            # Display to user
            pygame.display.flip()
        
        top_performers = game.get_top_ais(creatures//2)
        evolve(top_performers, creatures)

    # Quit pygame
    pygame.quit()