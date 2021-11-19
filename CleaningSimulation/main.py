"""To import the model in cellCleaning.py"""
from cellCleaning import CleaningModel

"""To animate the simulation"""
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
plt.rcParams["animation.html"] = "jshtml"
matplotlib.rcParams['animation.embed_limit'] = 2**128

"""To facilitate numeric operations"""
import numpy as np
import pandas as pd

"""To measure compilation time"""
import time
import datetime

"""Model constants"""
M = 10
N = 10
AGENTS = 3
DIRTY_PERCENTAGE = 9
MAX_TIME = 2.0

"""Start cronometer and initialize the model"""
start_time = time.time()
dirtyCells = int((DIRTY_PERCENTAGE/(M*N))*100)
model = CleaningModel(M, N, AGENTS, dirtyCells)
iterations = 0

"""Excecute a model step until there are no more dirty cells"""
while dirtyCells > 0 and time.time() - start_time < MAX_TIME:
    model.step()
    dirtyCells = model.dirty_cells
    iterations += 1

print('Tiempo de ejecución:', str(datetime.timedelta(seconds=(time.time() - start_time))))
print('Porcentaje de celdas limpias:', ((M*N-dirtyCells)/(M*N))*100)
print('Número de movimientos:', iterations*AGENTS)


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