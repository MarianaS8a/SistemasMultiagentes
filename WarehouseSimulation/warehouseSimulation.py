"""Mesa allows the use of agents and models"""
from matplotlib.pyplot import step
from mesa import Agent, Model
from mesa.time import SimultaneousActivation
from mesa.space import SingleGrid
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
        if model.boxGrid[x][y] > 0 or model.boxGrid[x][y] == -1:
            grid[x][y] = 1
    return grid

class OrganizingAgent(Agent):
    """Agent who sorts boxes"""
    def __init__(self, unique_id, model):
        """To initialize the agent with a unique_id and its respective model"""
        super().__init__(unique_id, model)
        self.hasBox = False

    def step(self):  
        """Finds out possible directions in which the agent can move,
        moves the agent to a random position takes or leaves a box depending on its hasBox status"""

        possible_steps = self.model.agentGrid.get_neighborhood(
            self.pos,
            moore=False,
            include_center=False)
        new_position = self.random.choice(possible_steps)

        neighbours = self.model.agentGrid.get_neighbors(new_position, moore=False, include_center=False)
        if len(neighbours) < 4:
            cellmates = self.model.agentGrid.get_cell_list_contents(new_position)
            if len(cellmates) == 0:
                self.model.agentGrid.move_agent(self, new_position)
                
                if self.hasBox == False and self.model.boxGrid[self.pos[0]][self.pos[1]] == 1:
                    self.model.boxGrid[self.pos[0]][self.pos[1]] = 0
                    self.hasBox = True

                elif self.hasBox == True and self.model.boxGrid[self.pos[0]][self.pos[1]] == -1:
                    self.model.boxGrid[self.pos[0]][self.pos[1]] = 2
                    self.hasBox = False
                    self.model.currentPiles -=1

                elif self.hasBox == True and 1 < self.model.boxGrid[self.pos[0]][self.pos[1]] < 5:
                    self.model.boxGrid[self.pos[0]][self.pos[1]] += 1
                    self.hasBox = False
                    self.model.currentPiles -=1
            else:
                self.step()

        
class WarehouseModel(Model):
    """The warehouse model manages each agent. Provides an agent grid which contains the position
     of each agent and a box grid which establishes the number of boxes in a cell"""
    def __init__(self, width, height, agents, boxes, targetPiles):
        """To initialize grids, agents and a data collector"""
        self.num_agents = agents
        self.agentGrid = SingleGrid(width, height, True)
        self.boxGrid = np.zeros( (self.agentGrid.width, self.agentGrid.height), dtype=float)
        self.schedule = SimultaneousActivation(self)
        self.boxes = boxes
        self.currentPiles = boxes
        self.targetPiles = targetPiles

        #Places random boxes
        # -1 if it is a place to pile boxes                                
        i = self.boxes-1
        k = targetPiles-1
        while(i >= 0):
            random1 = random.randint(0,self.agentGrid.width-1)
            random2 = random.randint(0,self.agentGrid.height-1)
            if self.boxGrid[random1][random2] == 0:
                self.boxGrid[random1][random2] = 1
                if k >= 0:
                    self.boxGrid[random1][random2] = -1
                    k -= 1                            
                i -= 1     

        #Initializes all agents in random empty coordinates
        j = agents-1
        while(j >= 0):
            random1 = random.randint(0,self.agentGrid.width-1)
            random2 = random.randint(0,self.agentGrid.height-1)
            cellmates = self.agentGrid.get_cell_list_contents([(random1,random2)])
            if self.boxGrid[random1][random2] == 0 and len(cellmates) == 0:
                a = OrganizingAgent(j, self)
                self.agentGrid.place_agent(a, (random1, random2))
                self.schedule.add(a)
                j = j - 1

        #Initializes a data collector and assigns get_grid as the collecting function
        self.datacollector = DataCollector(model_reporters={"Grid": get_grid})

    def step(self):
        """Collects the data of each iteration and orders the agents to move"""
        self.datacollector.collect(self)                                              
        self.schedule.step()
        if self.currentPiles == self.targetPiles:
            self.datacollector.collect(self)

