from matplotlib.pyplot import grid
from mesa import Agent, Model
from mesa.time import SimultaneousActivation
from mesa.space import MultiGrid
from mesa.space import SingleGrid
from mesa.datacollection import DataCollector
import random
import numpy as np
import pandas as pd

def get_grid(model):
    grid = np.zeros( (model.agentGrid.width, model.agentGrid.height) )
    for (content, x, y) in model.agentGrid.coord_iter():
        grid[x][y] = model.dirtyGrid[x][y]
    return grid

class CleaningAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)

    def step(self):  
        possible_steps = self.model.agentGrid.get_neighborhood(
            self.pos,
            moore=True,
            include_center=False)
        new_position = self.random.choice(possible_steps)
        self.model.agentGrid.move_agent(self, new_position)
        
        if self.model.dirtyGrid[self.pos[0]][self.pos[1]] == 1:
            self.model.dirtyGrid[self.pos[0]][self.pos[1]] = 0
            self.model.dirty_cells -= 1

class CleaningModel(Model):
    def __init__(self, width, height, agents, dirtyCells):
        self.num_agents = agents
        self.dirty_cells = dirtyCells
        self.agentGrid = MultiGrid(width, height, True)
        self.dirtyGrid = np.zeros( (self.agentGrid.width, self.agentGrid.height), dtype=float)
        self.schedule = SimultaneousActivation(self)

        for i in range (self.num_agents):
            a = CleaningAgent(i, self)
            self.agentGrid.place_agent(a, (1, 1))
            self.schedule.add(a)

        i = self.dirty_cells-1
        while(i >= 0):
            if self.dirtyGrid[random.randint(0,self.agentGrid.width-1)][random.randint(0,self.agentGrid.height-1)] == 0:
                self.dirtyGrid[random.randint(0,self.agentGrid.width-1)][random.randint(0,self.agentGrid.height-1)] = 1
                i = i - 1

        self.datacollector = DataCollector(model_reporters={"Grid": get_grid})

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()
        if self.dirty_cells == 0:
            self.datacollector.collect(self)

