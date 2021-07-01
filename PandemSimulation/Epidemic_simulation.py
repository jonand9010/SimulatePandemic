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


from networks.Networks import SIRNetwork
from simulation.NetworkSimulation import SIR_NetworkSimulation
from plotting.PandemPlots import WorldMapSimulation, SIRHistoryPlot


dir_path = os.path.dirname(os.path.realpath(__file__))
seed = 0
np.random.seed(seed)

datafiles = {'data': dir_path + '/data/graph_data.json', 'position':  dir_path + '/data/pos_dic.json'}
SIR_network = SIRNetwork(datafiles = datafiles, alpha = 0.1, beta = 0.8, gamma = 0.4)


SIR_simulation = SIR_NetworkSimulation(SIR_network, timesteps = 20)

SIR_simulation.simulate()

WorldMapSimulation(SIR_network, SIR_simulation).run_simulation()

SIRHistoryPlot(SIR_simulation)
