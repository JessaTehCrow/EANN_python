import pygame, random, time, math, json

from network import Network, random_network, evolve
from util import sigmoid
from typing import List


class Creature():
    def __init__(self, network:Network, thrust_power:float, map_size:list, gravity:float):
        self.nn = network

        self.size = map_size
        self.mid = [self.size[0]/2, self.size[1]/2]
        self.pos = list(self.mid)

        self.thrust = thrust_power
        self.gravity = gravity
        self.velocity = [0,0]

        self.survived_steps = 0
        self.dead = False
        self.fitness = 0

        self.spawn_margin = 80
        self.spawn_min = 20

        random_flip = False
        self.target = [self.mid[0]+(20*random.choice([-1,1][not random_flip:])), self.mid[1]-20]
        self.closest = 0
        self.hover_time = 0
        self.heighest_hover_time = 0
        self.collected = 0

        self.color = random.randint(0,16041983)
        self._hover_time = 50
        # self.new_target()


    # New target to collect
    def new_target(self):
        max = self.spawn_margin
        min = self.spawn_min
        ru = random.uniform
        self.target = [self.mid[0]+ru(-max,-max), self.mid[1]+ru(-max,max)]


    # Get the fitness of the AI
    def get_fitness(self):
        self.fitness = self.survived_steps/10 + self.closest*10 + self.hover_time*2 + self.heighest_hover_time*10 + self.collected*1000
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

        def distance(pos1,pos2):
            return sum(map(abs,[(y-x)/2 for x,y in zip(pos1,pos2)]))
        
        limited_pos = get_offset(self.pos)
        target = get_offset(self.target)

        old_target = [(self.target[0] - self.pos[0]), (self.target[1] - self.pos[1])]

        predictions = self.predict([*self.velocity, *old_target])

        x_thrust = predictions[1] - predictions[2]
        y_thrust = predictions[0]-0.1

        thrust = [x_thrust, y_thrust]

        forces = [thrust[0]*self.thrust[0], -(thrust[1]*self.thrust[1])+self.gravity]

        if predictions[3]>0.9:
            forces = [self.velocity[0]*-0.1,forces[1]/5]

        self.velocity = [(x/60)+y for x,y in zip(forces,self.velocity)]
        new_pos = [x+y for x,y in zip(self.pos, self.velocity)]

        target = get_offset(self.target)
        pos = get_offset(new_pos)
        dist = distance(target, pos)
        abs_distance = abs(dist)

        
        hover_distance = distance(get_offset([1,1]),get_offset([-1,-1]))
        if abs_distance <hover_distance and self.hover_time < self._hover_time:
            self.hover_time += 1
            self.closest = 1

        elif abs_distance <hover_distance and self.hover_time >= self._hover_time:
            self.collected += 1
            self.heighest_hover_time = 0
            self.hover_time = 0
            self.new_target()
        
        elif abs_distance >hover_distance and self.hover_time < self._hover_time:
            self.heighest_hover_time = self.hover_time
            self.hover_time = 0

        elif 1-abs_distance > self.closest:
            self.closest = 1-abs_distance
        
        if not (0<new_pos[0]<self.size[0] and 0<new_pos[1]<self.size[1]):
            self.dead = True
        self.pos = new_pos


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
    game_width = 300
    game_height = 200
    creature_size = 1
    target_size = 0.3
    max_iterations = 2000
    GTDR = 5 # Game to display ratio
    gravity = 3.6
    game = Game(game_width, game_height, gravity, max_iterations)

    # Initialize pygame
    pygame.init()
    screen = pygame.display.set_mode([game_width*GTDR, game_height*GTDR])
    font = pygame.font.SysFont("unispacebold", 25)
    RENDER = True

    # Define NN values
    inputs        = 4
    hidden        = [4, 8, 4]
    hidden_layers = len(hidden)
    outputs       = 4
    mutate_rate   = 0.05

    # Define evolve information
    creatures = 200
    generations = 2000000

    # Generate first generation
    ais = [random_network(inputs, hidden_layers, hidden, outputs) for ai in range(creatures)]
    averages = []

    for generation in range(generations):
        game.reset()
        game.load_network(ais, 1, 5)
        while not game.next():
            # pygame event handler
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    with open("fitness.json", 'w') as f:
                        json.dump(averages, f)

                    exit("USER EXIT")
                if event.type == pygame.KEYDOWN:
                    key = event.key
                    if key == pygame.K_r:
                        RENDER = not RENDER

            # Clear screen
            screen.fill(0x111111)

            def draw_text():
                text_draw = [f"Generation: {generation}",
                             f"Iteration: {game.iteration}", 
                             f"Dead: {game.dead}/{creatures}"]
                for i,text in enumerate(text_draw):
                    img = font.render(text, True, (0, 200, 255))
                    screen.blit(img,(10,10+(30*i)))
                
            if not RENDER:
                draw_text()
                pygame.display.flip()
                continue

            # Draw ais
            for ai in game.ais:
                pygame.draw.circle(screen, ai.color, [ai.pos[0]*GTDR, ai.pos[1]*GTDR], creature_size*GTDR)
                pygame.draw.circle(screen, ai.color, [ai.target[0]*GTDR, ai.target[1]*GTDR], target_size*GTDR)

            draw_text()
            # Display to user
            
            pygame.display.flip()
            # time.sleep(1/40)
            # time.sleep(1/25)
        
        top_performers = game.get_top_ais(creatures//2)
        if not generation%10:
            avg = sum([ai.fitness for ai in game.ais])/len(top_performers)
            averages.append(avg)
        ais = evolve(top_performers, creatures, mutate_rate, True)

    # Quit pygame
    pygame.quit()