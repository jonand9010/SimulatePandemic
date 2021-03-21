# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 16:17:21 2020

@author: joanxa
"""

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt



def make_nodes(nodes, layers):
    x = np.ones((nodes, layers))
    y = np.zeros((nodes, layers))
    for l in range(layers):
        if l == 0:
            x[:, l] = np.ones(x[:,0].shape)
        else:
            x[:, l] = x[:, l-1] + 1
        for i in range(nodes):
            y[i, l] = i
    
    data = {'x': x, 'y': y}
    return(data)



def draw_edges(x, y):
    x = data['x']
    y = data['y']

    nodes = len(x[:,0])
    layers = len(x[0,:])
    
    for l in range(layers):
        for n1 in range(nodes):
            p_from = np.array([x[n1, l], y[n1, l]])
            
            for n2 in range(nodes):
                if l < layers-1:
                    p_to = np.array([x[n2, l+1], y[n2, l+1]])
                    plt.plot([p_from[0], p_to[0]], [p_from[1], p_to[1]], 'k-', linewidth=0.3, alpha = 0.3, zorder=-1)
    
        plt.scatter(x[:,l], y[:,l], color = '#1f77b4')
    plt.xlim([0, layers+1])
    
    
def draw_network(network, save = False):
    fig, ax = plt.subplots()
    layers = len(network)
    x = {}
    y = {}
    for l in range(layers):

        L = 'l' + str(l)
        if l == 0:
            x[L] = np.ones((network[l], 1))
        else:
            x[L] = np.ones((network[l], 1))*x['l' + str(l-1)][0] + 1
            
        y[L] = np.zeros((network[l], 1)) + network[l] / 2
        for i in range(network[l]):
            y[L][i] = y[L][i] - i
    
    for l in range(layers):     
        
        L = 'l' + str(l)
        
        for n1 in range(network[l]):
            p_from = np.array([x[L][n1], y[L][n1]])
            if l < layers-1:
                L_next = 'l' + str(l+1)
                for n2 in range(network[l+1]):
                    p_to = np.array([x[L_next][n2], y[L_next][n2]])
                    plt.plot([p_from[0], p_to[0]], [p_from[1], p_to[1]], 'k-', linewidth=0.3, alpha = 0.3, zorder=-1)
        plt.scatter(x[L], y[L], color = '#1f77b4')
    plt.xlim([0, layers+1])
    ax.axis('off')
    if save:
        plt.savefig('Neuralnetwork.png', transparent=True)
    
    return x, y


x, y = draw_network([20, 15, 15, 10, 5], True)