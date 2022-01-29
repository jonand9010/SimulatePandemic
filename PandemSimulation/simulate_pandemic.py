#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%%
"""
Created on Sat Mar 14 13:28:30 2020

@author: Jonas
"""
#%%

import numpy as np
import json
from pathlib import Path
import os

from networks.SIR import SIRNetwork
from simulation.NetworkSimulation import SIR_NetworkSimulation
from plotting.PandemPlots import WorldMapSimulation, SIRHistoryPlot


dir_path = os.path.dirname(os.path.realpath(__file__))
seed = 0
np.random.seed(seed)

datafiles = {'data': dir_path + '/data/graph_data.json', 'position':  dir_path + '/data/pos_dic.json'}
SIR_network = SIRNetwork(datafiles = datafiles, travel_rate = 0.01, travel_infection_rate = 2, beta = 1, gamma = 0.4)

start_parameters = {'city': 'WUH', 'infected': 100, 'node_populations': 10000 * np.ones((len(SIR_network.nodes), ))}
SIR_simulation = SIR_NetworkSimulation(SIR_network, timesteps = 100, start_parameters = start_parameters)

SIR_simulation.simulate()

WorldMapSimulation(SIR_network, SIR_simulation).run_simulation()

SIRHistoryPlot(SIR_simulation)
