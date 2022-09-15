import random

from util.network import Network
from util.funcs import sigmoid
from typing import List


class Creature():
    def __init__(self, network:Network, thrust_power:float, map_size:list, gravity:float, generation:int, max_dots:int):
        self.nn = network
        self.generation = generation

        self.size = map_size
        self.mid = [self.size[0]/2, self.size[1]/2]
        self.pos = list(self.mid)

        self.thrust = thrust_power
        self.gravity = gravity
        self.velocity = [0,0]

        self.survived_steps = 0
        self.dead = False
        self.fitness = 0
        self.max_dots = max_dots

        self.spawn_distance = [50, 30]
        self.center_offset = 20

        random_flip = True
        random.seed(self.generation)
        flipx = random.choice([-1,1]) if random_flip else 1
        # flipy = random.choice([-1,1]) if random_flip else 1
        flipy = -1

        self.target = [self.mid[0]+(20*flipx), self.mid[1]-(20*flipy)]
        self.closest = 0
        self.hover_time = 0
        self.heighest_hover_time = 0
        self.collected = 0
        self._hover_time = 50
        self.new_color()
       
        # self.new_target()

    def new_color(self):
        random.seed(self.generation + self.collected)
        self.color = random.randint(0,16041983) 
    
    # New target to collect
    def new_target(self):
        if self.collected >= self.max_dots:
            self.dead = True
            return

        random.seed(self.generation + self.collected)
        pos1 = random.uniform(-self.spawn_distance[0], self.spawn_distance[0])
        pos2 = random.uniform(-self.spawn_distance[1], self.spawn_distance[1])
            
        self.target = [self.mid[0]+pos1, self.mid[1]+pos2]


    # Get the fitness of the AI
    def get_fitness(self):
        self.fitness = self.closest*100 + self.hover_time*2 + self.heighest_hover_time*10 + self.collected*1000
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
        
        target = get_offset(self.target)

        old_target = [(self.target[0] - self.pos[0]), (self.target[1] - self.pos[1])]

        predictions = self.predict([sigmoid(self.velocity[0]), sigmoid(self.velocity[1]), *old_target])

        x_thrust = predictions[1] - predictions[2]
        y_thrust = predictions[0]-0.1

        thrust = [x_thrust, y_thrust]

        forces = [thrust[0]*self.thrust[0], -(thrust[1]*self.thrust[1])+self.gravity]

        # if predictions[3]>0.95:
        #     forces = [self.velocity[0]*-0.5,self.velocity[1]*-0.5]

        self.velocity = [(x/60)+y for x,y in zip(forces,self.velocity)]
        new_pos = [x+y for x,y in zip(self.pos, self.velocity)]

        target = get_offset(self.target)
        pos = get_offset(new_pos)
        dist = distance(target, pos)
        abs_distance = abs(dist)
        self.get_fitness()

        
        hover_distance = distance(get_offset([0,0]),get_offset([1,1]))
        if abs_distance <hover_distance and self.hover_time < self._hover_time:
            self.hover_time += 1
            self.closest = 1

        elif abs_distance <hover_distance and self.hover_time >= self._hover_time:
            self.collected += 1
            self.heighest_hover_time = 0
            self.hover_time = 0
            self.new_target()
            self.new_color()
        
        elif abs_distance >hover_distance and self.hover_time < self._hover_time:
            self.heighest_hover_time = self.hover_time
            self.hover_time = 0

        if 1-abs_distance > self.closest:
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
        self.max_dots = 9999
    
    
    # Add network to game data
    def register_network(self, ai:Network, thrust_power_up:float, thrust_power_side:float, generation:int):
        self.ais.append( Creature(ai, [thrust_power_side, thrust_power_up], [self.width,self.height], self.gravity, generation, self.max_dots))
    

    def load_network(self, ais:List[Network], thrust_power_side:float, thrust_power_up:float, generation:int) -> None:
        for ai in ais:
            self.register_network(ai, thrust_power_up, thrust_power_side, generation)
    

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