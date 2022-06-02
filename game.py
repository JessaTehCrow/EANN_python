from network import Network, random_network
import pygame

class Creature():
    def __init__(self, network:Network, thrust_power:float, map_size:list):
        self.nn = network

        self.pos = [0,0]
        self.size = map_size

        self.thrust = thrust_power
        self.forces = [0, 0]
        self.velocity = [0, 0]

        self.survived_steps = 0

    
    # Get the fitness of the AI
    def fitness(self):
        return self.survived_steps**1.1//1
    

    # Predict solution from senario input
    def predict(self, inputs:list):
        return self.nn.predict(inputs)
    

    # Next step
    def next(self):
        self.forces = self.predict()


class Game():
    def __init__(self, width:int, height:int, gravity:float):
        self.width = width
        self.height = height
        self.gravity = gravity
        self.ais = []
    
    
    # Add network to game data
    def register_network(self, ai:Network):
        self.ais.append( Creature(ai), [self.width,self.height] )
    

    # Next step for all registered AIs
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