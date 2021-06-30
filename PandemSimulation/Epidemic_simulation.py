#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%%
"""
Created on Sat Mar 14 13:28:30 2020

@author: Jonas
"""
#%%

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
import json
from pathlib import Path
import os


from Network.Networks import SIRNetwork
from Simulation.NetworkSimulation import SIR_NetworkSimulation


#%%

dir_path = os.path.dirname(os.path.realpath(__file__))
seed = 0
np.random.seed(seed)

datafiles = {'data': dir_path + '/data/graph_data.json', 'position':  dir_path + '/data/pos_dic.json'}
SIR_network = SIRNetwork(datafiles = datafiles, alpha = 0.1, beta = 0.8, gamma = 0.4)


SIR_simulation = SIR_NetworkSimulation(SIR_network, timesteps = 120)

SIR_simulation.simulate(SIR_network)

#%%
fig1 = plt.figure()
import cartopy.crs as ccrs
ax1 = plt.subplot(111, projection = ccrs.PlateCarree())
sns.set()
#plt.close('all')

for t in range(SIR_simulation.timesteps):

    ax1.stock_img()
    ax1.coastlines()
    nx.draw_networkx(SIR_network.Graph, ax = ax1, font_size=10,
             alpha=.25,
             width=.1,
             node_size=0.1*SIR_simulation.I[:,t],
             with_labels=False,
             pos=SIR_network.pos,
             node_color = 'r',
             cmap=plt.cm.autumn)

    if t != SIR_simulation.timesteps-1:
        if SIR_simulation.SIR[1, t] == 0:
            break
        
        plt.pause(0.1)
        ax1.cla()
#%%

fig2 = plt.figure()
ax2 = plt.subplot(111)
l1, = ax2.plot(SIR_simulation.SIR[0]/SIR_simulation.SIR[0,0], 'b', alpha = 0.5)
l2, = ax2.plot(SIR_simulation.SIR[1]/SIR_simulation.SIR[0,0], 'r', alpha = 0.5)
l3, = ax2.plot(SIR_simulation.SIR[2]/SIR_simulation.SIR[0,0], 'g', alpha = 0.5)
ax2.legend([l1, l2, l3], ['Susceptible', 'Infected', 'Recovered'])
ax2.set_title('R0: ' + str(SIR_network.R0))
ax2.set_xlabel('Days')
plt.show()
    #plt.savefig('Epidemic' + str(t) + '.png')

 
# %%