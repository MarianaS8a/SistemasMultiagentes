"""To import the model in cellCleaning.py"""
from warehouseSimulation import WarehouseModel

"""To animate the simulation"""
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
plt.rcParams["animation.html"] = "jshtml"
matplotlib.rcParams['animation.embed_limit'] = 2**128

"""To facilitate numeric operations"""
import numpy as np
import pandas as pd
from math import ceil

"""To measure compilation time"""
import time
import datetime

"""Model constants"""
M = 10
N = 10
AGENTS = 3
BOXES = 9
MAX_TIME = 10.0

"""Start cronometer and initialize the model"""
start_time = time.time()
targetPiles = ceil(BOXES/5)
model = WarehouseModel(M, N, AGENTS, BOXES, targetPiles)
iterations = 0
currentPiles = model.currentPiles

"""Excecute a model step until there are only boxes/5 piles"""
while currentPiles > targetPiles and time.time() - start_time < MAX_TIME:
    model.step()
    currentPiles = model.currentPiles
    iterations += 1

print('Tiempo de ejecución:', str(datetime.timedelta(seconds=(time.time() - start_time))))
print('Número de movimientos:', iterations*AGENTS)
print('Pilas finales:', currentPiles)


"""Model animation"""
all_grid = model.datacollector.get_model_vars_dataframe()

fig, axs = plt.subplots(figsize=(7,7))
axs.set_xticks([])
axs.set_yticks([])
patch = plt.imshow(all_grid.iloc[0][0], cmap=plt.cm.binary)

def animate(i):
    patch.set_data(all_grid.iloc[i][0])

anim = animation.FuncAnimation(fig, animate, frames=iterations+1)
plt.show()