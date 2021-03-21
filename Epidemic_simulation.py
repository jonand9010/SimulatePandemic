#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 13:28:30 2020

@author: Jonas
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
import json



def get_graph(n_nodes = 10, clusters = 5):
    choice = np.zeros((clusters, 1))
    choice[0] = 1
    pos = {}
    for i in range(n_nodes):
        r = np.random.permutation(choice)
        for c in range(clusters):
            gauss = r[c]*np.random.normal([c,c], 1.5, 2)
            #gauss = r[0]*np.random.normal([0,0], 1.5, 2) + r[1]*np.random.normal([-5, 5], 1.5, 2) \
            #+ r[2]*np.random.normal([5, 5], 1.5, 2)  
        pos[i] = gauss
    G = nx.random_geometric_graph(n_nodes, radius = c, pos=pos)
    return G, pos


def Laplacian(A): 
    #A = A - 2*np.tril(A)    # Changes the sign of lower triangle of the matrix (to account for in/out flux)
    I = np.eye(A.shape[0])      #Adding self loop
    D = np.sum(A, axis = 0)     #Calculated the number of edges per node, i.e. the degree
    D[D == 0] = np.inf          #Defines 0 degree as infinite to avoid dividing with zero later
    D_norm = np.sqrt(1/D)       #Inverse square root for normalization
    D_norm = np.diag(D_norm)    #Normalization matrix
    
    L = I - np.dot(D_norm, np.dot( A, D_norm))
    L = np.nan_to_num(L)
    return L

def get_infected_travellers(n_nodes, N, I, dNdt):
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


def update_infected_adjacency_matrix(n_nodes, A, Flux_I):


    while Flux_I > 0:
        edge_list = np.where(A[n,:] == 1)[0]     #List of non-zero edges
        random_edge = np.random.choice(edge_list)   #Random choice of edge
        Nr_I  = np.random.randint(0, Flux_I+1) #Number of random infected to move in a certain direction
        A_I[random_edge, n] = Nr_I  #Nr_I infected moving along edge random_edge 
        A_I[n, n] = -Nr_I
        Flux_I = Flux_I - Nr_I #Count down the total number of infected left to distribute

                

#%%
sns.set()
seed = 0
np.random.seed(seed)

T = 120 # Max number of days of simulation

beta = .8  #transmission rate
gamma = 0.4 #recovery rate
public_trans = 0.1  # alpha
R0 = beta/gamma

'''
n_nodes = 10 

G = nx.random_geometric_graph(n_nodes, 0.2, seed=seed) # Random geometric network of n_nodes cities
#pos = nx.get_node_attributes(G, 'pos')  #Save the node positions


'''


with open('graph_data.json') as json_file:
    import_data = json.load(json_file)
with open('pos_dic.json') as json_file:
    import_pos = json.load(json_file)



import_graph = nx.node_link_graph(import_data)


G = import_graph
pos = import_pos
n_nodes = nx.number_of_nodes(G)
A = nx.to_numpy_array(G)    #Save the adjacency matrix of the network


L_sym = Laplacian(A)    # Calculate graph Laplacian


N = np.zeros((n_nodes, T), dtype = 'int')   #Initialize vector for city populations
S, I, R = N.copy(), N.copy(), N.copy()      #Initialize vector for susceptible, infected, and recovered in each city

#N[:,0] = np.random.randint(100,1000000, size = n_nodes)    #Random population in each city at t=0
N[:,0] = 10000 * np.ones((n_nodes, ))
#r_node = np.random.randint(n_nodes)                     #Randomly picked city to be infected

list_pos = list(pos.keys())
pos_list = list(pos.items())
pos_new = {}
for i in range(len(G.nodes())):
    #pos_new[i] = pos_list[i][1]
    if list(G.nodes())[i] == 'WUH':
        start_pos = i

r_node = start_pos
I[r_node,0] = np.random.randint(10)          #Random number of infected in city r_node
S[:,0] = N[:,0] - I[:,0]        # Defining the number of susceptible in each city at t=0



SIR = np.zeros(shape = (3, T))  #Initialize total SIR matrix
SIR[:, 0] = np.sum(S[:,0]), np.sum(I[:,0]), np.sum(R[:,0])


for t in range(T-1):
    dNdt = - public_trans * np.dot(L_sym, N[:, t]) # Number of people travelling to another city each day

    #dNdt = dNdt - np.sum(dNdt) / n_nodes # Making sure that the system is closed
    dNdt = np.round(dNdt)

    #N[:, t+1] = N[:,t] + dNdt
    A_I = np.zeros((n_nodes, n_nodes))  #Initializing adjacency matrix for infected
    
    
    for n in range(n_nodes):

        Flux_I = get_infected_travellers(n_nodes, N[n, t], I[n, t], dNdt[n])

        
        update_infected_adjacency_matrix(n_nodes, A, Flux_I)
      
        
    #Correction from movements
    dI = np.sum(A_I, axis = 1)
    #if t < 100:
        #print(np.sum(dI))

    I[:, t] = I[:, t] + dI 
    S[:, t] = S[:, t] - dI
    

    dSdt = -beta * I[:, t] * S[:, t] / N[:, t]
    dIdt = beta * I[:, t] * S[:, t] / N[:, t] - gamma * I[:, t]
    dRdt = gamma*I[:, t]
    
    S[:, t+1] = S[:, t] + dSdt
    I[:, t+1] = I[:, t] + dIdt
    R[:, t+1] = R[:, t] + dRdt
    N[:, t+1] = S[:, t+1] + I[:, t+1] + R[:, t+1]

    SIR[:, t+1] = np.sum(S[:,t+1]), np.sum(I[:,t+1]), np.sum(R[:,t+1])

N[:, T-1] = S[:, T-1] + I[:, T-1] + R[:, T-1]


#%%
import cartopy.crs as ccrs
plt.close('all')
crs = ccrs.PlateCarree()

fig1 = plt.figure()
ax1 = plt.subplot(111, projection = crs)

for t in range(T):

    ax1.stock_img()
    nx.draw_networkx(G, ax = ax1, font_size=10,
             alpha=.25,
             width=.1,
             node_size=0.1*I[:,t],
             with_labels=False,
             pos=pos,
             node_color = 'r',
             cmap=plt.cm.autumn)

    if t != T-1:
        if SIR[1, t] == 0:
            break
        
        plt.pause(0.1)
        ax1.cla()
#%%
fig2 = plt.figure()
ax2 = plt.subplot(111)
l1, = ax2.plot(SIR[0]/SIR[0,0], 'b', alpha = 0.5)
l2, = ax2.plot(SIR[1]/SIR[0,0], 'r', alpha = 0.5)
l3, = ax2.plot(SIR[2]/SIR[0,0], 'g', alpha = 0.5)
ax2.legend([l1, l2, l3], ['Susceptible', 'Infected', 'Recovered'])
ax2.set_title('R0: ' + str(R0))
ax2.set_xlabel('Days')
    #plt.savefig('Epidemic' + str(t) + '.png')

 