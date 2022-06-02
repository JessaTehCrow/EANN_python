from network import Network, random_network
import pygame

class Creature():
    def __init__(self, network:Network):
        self.nn = network
        self.pos = [0,0]

        self.forces = [0, 0]
        self.velocity = [0, 0]
    

    def predict(self, inputs:list):
        return self.nn.predict(inputs)
    

    def next(self):
        pass


class Game():
    def __init__(self, width:int, height:int, gravity:float):
        self.width = width
        self.height = height
        self.gravity = gravity
        self.ais = []
    
    
    def register_network(self, ai:Network):
        self.ais.append(ai)
    

    def next(self):
        pass


if __name__ == "__main__":
    # Game information
    game_width = 150
    game_height = 100
    creature_size = 1
    GTDR = 7 # Game to display ratio
    gravity = 3.6

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

    # Quit pygame
    pygame.quit()