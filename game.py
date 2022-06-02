import pygame, random, time

from network import Network, random_network, evolve
from util import sigmoid
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
        self.fitness = 0

        self.target = [0,0]
        self.closest = 0
        self.collected = 0

        self.color = random.randint(0,16041983)
        self.new_target()


    # New target to collect
    def new_target(self):
        self.target = [random.uniform(10,self.size[0]-10), random.uniform(10,self.size[1]-10)]


    # Get the fitness of the AI
    def get_fitness(self):
        self.fitness = self.closest + self.collected*1000
        return self.fitness
    

    # Predict solution from senario input
    def predict(self, inputs:list):
        return self.nn.predict(inputs)
    

    # Next step
    def next(self):
        def get_offset(target:list):
            x_off = (self.size[0]/2 - target[0])*2/self.size[0]
            y_off = (self.size[1]/2 - target[1])*2/self.size[1]
            return [x_off, y_off]
        
        limited_pos = get_offset(self.pos)
        target = get_offset(self.target)

        limited_velocity = [sigmoid(x) for x in self.velocity]
        old_target = [(y-x)/2 for x,y in zip(target, limited_pos)]

        predictions = self.predict([*limited_pos, *limited_velocity, *old_target])
        thrust = [predictions[1]-predictions[2], predictions[0]]

        forces = [thrust[0]*self.thrust[0], -(thrust[1]*self.thrust[1])+self.gravity]

        self.velocity = [(x/60)+y for x,y in zip(forces,self.velocity)]
        self.pos = [x+y for x,y in zip(self.pos, self.velocity)]

        target = get_offset(self.target)
        pos = get_offset(self.pos)
        distance = sum(map(abs,[(y-x)/2 for x,y in zip(target,pos)]))

        if abs(distance) <0.025:
            self.collected += 1
            self.closest = 1
            self.new_target()

        elif 1-abs(distance) > self.closest:
            self.closest = 1-abs(distance)
        
        if not (0<self.pos[0]<self.size[0] and 0<self.pos[1]<self.size[1]):
            self.dead = True


class Game():
    def __init__(self, width:int, height:int, gravity:float, max_iterations:int):
        self.width = width
        self.height = height
        self.gravity = gravity
        self.ais:List[Creature] = []
        self.max_iters = max_iterations
        self.iteration = 0
        self.dead = 0
    
    
    # Add network to game data
    def register_network(self, ai:Network, thrust_power_up:float, thrust_power_side:float):
        self.ais.append( Creature(ai, [thrust_power_side, thrust_power_up], [self.width,self.height], self.gravity) )
    

    def load_network(self, ais:List[Network], thrust_power_side:float, thrust_power_up:float) -> None:
        for ai in ais:
            self.register_network(ai, thrust_power_up, thrust_power_side)
    

    # Next step for all registered AIs
    def next(self):
        self.iteration += 1
        for ai in self.ais:
            if ai.dead:
                continue
            ai.next()

        dead = [ai.dead for ai in self.ais]
        self.dead = sum(dead)
        return all(dead) or (self.iteration > self.max_iters)
    

    # Remove all saved networks
    def reset(self):
        self.ais = []
        self.iteration = 0
    

    # Get top ais based on their fitness
    def get_top_ais(self, cut_off:int):
        top = sorted(self.ais, key=lambda ai: ai.get_fitness(), reverse=True)[:cut_off]
        return top


if __name__ == "__main__":
    # Game information
    game_width = 150
    game_height = 100
    creature_size = 1
    target_size = 0.5
    max_iterations = 2000
    GTDR = 7 # Game to display ratio
    gravity = 3.6
    game = Game(game_width, game_height, gravity, max_iterations)

    # Initialize pygame
    pygame.init()
    screen = pygame.display.set_mode([game_width*GTDR, game_height*GTDR])
    font = pygame.font.SysFont("unispacebold", 25)

    # Define NN values
    inputs        = 6
    hidden_layers = 2
    hidden        = 4
    outputs       = 4
    mutate_rate   = 0.03

    # Define evolve information
    creatures = 400
    generations = 1000

    # Generate first generation
    ais = [random_network(inputs, hidden_layers, hidden, outputs) for ai in range(creatures)]

    for generation in range(generations):
        game.reset()
        game.load_network(ais, 2, 5)
        while not game.next():
            # pygame event handler
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    exit("USER EXIT")

            # Clear screen
            screen.fill(0x111111)

            # Draw ais
            for ai in game.ais:
                pygame.draw.circle(screen, ai.color, [ai.pos[0]*GTDR, ai.pos[1]*GTDR], creature_size*GTDR)
                pygame.draw.circle(screen, ai.color, [ai.target[0]*GTDR, ai.target[1]*GTDR], target_size*GTDR)

            # Display to user
            text_draw = [f"Generation: {generation}",f"Iteration: {game.iteration}", f"Dead: {game.dead}/{creatures}"]
            for i,text in enumerate(text_draw):
                img = font.render(text, True, (0, 200, 255))
                screen.blit(img,(10,10+(30*i)))
            pygame.display.flip()
            # time.sleep(1/40)
            # time.sleep(1/25)
        
        top_performers = game.get_top_ais(creatures//2)
        ais = evolve(top_performers, creatures, mutate_rate, True)

    # Quit pygame
    pygame.quit()