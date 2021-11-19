

"""Mesa allows the use of agents and models"""
from mesa import Agent, Model
from mesa.time import SimultaneousActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector

"""To generate random numbers"""
import random

"""To facilitate numeric operations"""
import numpy as np
import pandas as pd

def get_grid(model):
    """Generates a copy of a grid"""
    grid = np.zeros( (model.agentGrid.width, model.agentGrid.height) )
    for (content, x, y) in model.agentGrid.coord_iter():
        grid[x][y] = model.dirtyGrid[x][y]
    return grid

class CleaningAgent(Agent):
    """Agent who cleans the dirtyGrid"""
    def __init__(self, unique_id, model):
        """To initialize the agent with a unique_id and its respective model"""
        super().__init__(unique_id, model)

    def step(self):  
        """Finds out possible directions in which the agent can move,
        moves the agent to a random position and cleans the new cell"""
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
    """The cleaning model manages each agent. Provides an agent grid which contains the position
     of each agent and a dirty grid which establishes if a cell is clean or dirty"""
    def __init__(self, width, height, agents, dirtyCells):
        """To initialize grids, agents and a data collector"""
        self.num_agents = agents
        self.dirty_cells = dirtyCells
        self.agentGrid = MultiGrid(width, height, True)
        self.dirtyGrid = np.zeros( (self.agentGrid.width, self.agentGrid.height), dtype=float)
        self.schedule = SimultaneousActivation(self)

        #Initializes all agents in the coordinate 1,1 of the agent grid
        for i in range (self.num_agents):
            a = CleaningAgent(i, self)
            self.agentGrid.place_agent(a, (1, 1))
            self.schedule.add(a)

        #Defines random cells as dirty                                 
        i = self.dirty_cells-1   
        while(i >= 0):
            if self.dirtyGrid[random.randint(0,self.agentGrid.width-1)][random.randint(0,self.agentGrid.height-1)] == 0:
                self.dirtyGrid[random.randint(0,self.agentGrid.width-1)][random.randint(0,self.agentGrid.height-1)] = 1
                i = i - 1

        #Initializes a data collector and assigns get_grid as the collecting function
        self.datacollector = DataCollector(model_reporters={"Grid": get_grid})

    def step(self):
        """Collects the data of each iteration and orders the agents to move"""
        self.datacollector.collect(self)                                              
        self.schedule.step()
        if self.dirty_cells == 0:
            self.datacollector.collect(self)

