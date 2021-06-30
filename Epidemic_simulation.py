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

class SIRNetwork:

    def __init__(self, datafiles, simulated_days, alpha, beta, gamma):
        self.Graph, self.A, self.pos = self.load_graph(datafiles['data'], datafiles['position'])
        self.Number_of_nodes = self.A.shape[0]
        self.L_sym = self.Laplacian(self.A)    # Calculate symmetric graph Laplacian
        self.simulated_days = simulated_days
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.R0 = self.beta/self.gamma

        #self.init_simulation(self.Number_of_nodes)

    def load_graph(self, file_graph, file_positions):
        
        with open(file_graph) as json_file:
            import_data = json.load(json_file)
        with open(file_positions) as json_file:
            import_pos = json.load(json_file)

        import_graph = nx.node_link_graph(import_data)

        G = import_graph
        pos = import_pos
        n_nodes = nx.number_of_nodes(G)
        A = nx.to_numpy_array(G)    #Save the adjacency matrix of the network
        
        return G, A, pos


    def Laplacian(self, A): 

        I = np.eye(A.shape[0])      #Adding self loop
        D = np.sum(A, axis = 0)     #Calculated the number of edges per node, i.e. the degree
        D[D == 0] = np.inf          #Defines 0 degree as infinite to avoid dividing with zero later
        D_norm = np.sqrt(1/D)       #Inverse square root for normalization
        D_norm = np.diag(D_norm)    #Normalization matrix
        
        L = I - np.dot(D_norm, np.dot( A, D_norm))
        L = np.nan_to_num(L)
        return L
class NetworkSimulation:

    def __init__(self, Network, timesteps):

        self.N = np.zeros((Network.Number_of_nodes, timesteps), dtype = 'int')   #Initialize vector for city populations
        self.S, self.I, self.R = self.N.copy(), self.N.copy(), self.N.copy()      #Initialize vector for susceptible, infected, and recovered in each city
        self.A_I = np.zeros((Network.Number_of_nodes, Network.Number_of_nodes))  #Initializing adjacency matrix for infected
        self.SIR = np.zeros((3, timesteps))
        
        self.N[:,0] = 10000 * np.ones((self.N.shape[0], )) #Population in each city at t=0
        
        list_pos = list(Network.pos.keys())
        pos_list = list(Network.pos.items())

        for i in range(Network.Number_of_nodes):

            if list(Network.Graph.nodes())[i] == 'WUH': #Selecting Wuhan as first infected city
                start_pos = i

        r_node = start_pos
        self.I[r_node,0] = np.random.randint(10)          #Random number of infected in city r_node
        self.S[:,0] = self.N[:,0] - self.I[:,0]        # Defining the number of susceptible in each city at t=0



        self.SIR = np.zeros(shape = (3, Network.simulated_days))  #Initialize total SIR matrix
        self.SIR[:, 0] = np.sum(self.S[:,0]), np.sum(self.I[:,0]), np.sum(self.R[:,0])

    def get_infected_travellers(self, n_nodes, N, I, dNdt):
        if N == 0:
            I = 0
            p = 0         #Probability that a traveller is infected
        elif I >= N:
            I = N
            p = 1
        elif I < 0:
            #print('Warning I < 0')
            I = 0
            p = 0
        else:
            p = I/N

        
        n = np.int(np.abs(dNdt)) #Number of travellers to choose from

        Flux_I = np.random.binomial(n , p)  #The number of infected travellers
            
        return Flux_I


    def update_infected_adjacency_matrix(self, n_nodes, A, n, Flux_I):


        while Flux_I > 0:
            edge_list = np.where(A[n,:] == 1)[0]     #List of non-zero edges
            random_edge = np.random.choice(edge_list)   #Random choice of edge
            Nr_I  = np.random.randint(0, Flux_I+1) #Number of random infected to move in a certain direction
            self.A_I[random_edge, n] = Nr_I  #Nr_I infected moving along edge random_edge 
            self.A_I[n, n] = -Nr_I
            Flux_I = Flux_I - Nr_I #Count down the total number of infected left to distribute

    def simulate(self, Network):
        for t in range(Network.simulated_days-1):
            dNdt = - Network.alpha * np.dot(Network.L_sym, self.N[:, t]) # Number of people travelling to another city each day

            dNdt = np.round(dNdt)

            print(f"Timestep: {t+1} of { Network.simulated_days-1}")
    
            for n in range(Network.Number_of_nodes):

                Flux_I = self.get_infected_travellers(Network.Number_of_nodes, self.N[n, t], self.I[n, t], dNdt[n])
                
                self.update_infected_adjacency_matrix(Network.Number_of_nodes, Network.A, n, Flux_I)
            
            #Correction from movements
            dI = np.sum(self.A_I, axis = 1)

            self.I[:, t] = self.I[:, t] + dI 
            self.S[:, t] = self.S[:, t] - dI
            
            dSdt = -Network.beta * self.I[:, t] * self.S[:, t] / self.N[:, t]
            dIdt = Network.beta * self.I[:, t] * self.S[:, t] / self.N[:, t] - Network.gamma * self.I[:, t]
            dRdt = Network.gamma*self.I[:, t]
            
            self.S[:, t+1] = self.S[:, t] + dSdt
            self.I[:, t+1] = self.I[:, t] + dIdt
            self.R[:, t+1] = self.R[:, t] + dRdt
            self.N[:, t+1] = self.S[:, t+1] + self.I[:, t+1] + self.R[:, t+1]

            self.SIR[:, t+1] = np.sum(self.S[:,t+1]), np.sum(self.I[:,t+1]), np.sum(self.R[:,t+1])

        self.N[:, Network.simulated_days-1] = self.S[:, Network.simulated_days-1] + self.I[:, Network.simulated_days-1] + self.R[:, Network.simulated_days-1]

#%%

seed = 0
np.random.seed(seed)

datafiles = {'data': 'graph_data.json', 'position':  'pos_dic.json'}
SIR_network = SIRNetwork(datafiles = datafiles, simulated_days = 30, alpha = 0.1, beta = 0.8, gamma = 0.4)


SIR_simulation = NetworkSimulation(SIR_network, 30)

SIR_simulation.simulate(SIR_network)

#%%
fig1 = plt.figure()
ax1 = plt.subplot(111, projection = crs)
import cartopy.crs as ccrs
sns.set()
plt.close('all')

for t in range(SIR_network.simulated_days):

    ax1.stock_img()
    nx.draw_networkx(SIR_network.Graph, ax = ax1, font_size=10,


             alpha=.25,
             width=.1,
             node_size=0.1*SIR_simulation.I[:,t],
             with_labels=False,
             pos=SIR_network.pos,
             node_color = 'r',
             cmap=plt.cm.autumn)

    if t != SIR_network.simulated_days-1:
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