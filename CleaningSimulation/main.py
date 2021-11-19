from cellCleaning import CleaningModel

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
plt.rcParams["animation.html"] = "jshtml"
matplotlib.rcParams['animation.embed_limit'] = 2**128

import numpy as np
import pandas as pd


import time
import datetime

M = 10
N = 10
AGENTS = 3
DIRTY_PERCENTAGE = 9
MAX_TIME = 2.0

start_time = time.time()
dirtyCells = int((DIRTY_PERCENTAGE/(M*N))*100)
model = CleaningModel(M, N, AGENTS, dirtyCells)
iterations = 0

while dirtyCells > 0:
    model.step()
    dirtyCells = model.dirty_cells
    iterations += 1

print('Tiempo de ejecución:', str(datetime.timedelta(seconds=(time.time() - start_time))))
print('Porcentaje de celdas limpias:', ((M*N-dirtyCells)/(M*N))*100)
print('Número de movimientos:', iterations*AGENTS)

all_grid = model.datacollector.get_model_vars_dataframe()

fig, axs = plt.subplots(figsize=(7,7))
axs.set_xticks([])
axs.set_yticks([])
patch = plt.imshow(all_grid.iloc[0][0], cmap=plt.cm.binary)

def animate(i):
    patch.set_data(all_grid.iloc[i][0])

anim = animation.FuncAnimation(fig, animate, frames=iterations+1)
plt.show()