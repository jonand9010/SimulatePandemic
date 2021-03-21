# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 17:00:06 2020

@author: joanxa
"""

#%%

import math
import json
import numpy as np
import pandas as pd
import networkx as nx
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from IPython.display import Image
from mayavi import mlab
from mpl_toolkits.mplot3d import Axes3D


names = ('id','name','x','y','z','left/right', '.')
data = pd.read_csv('desikan_atlas.txt', header=0, delimiter='\t', names = names)

pos = {data: (v['x'], v['y'], v['z'])
       for data, v in
       data.to_dict('index').items()}

N = len(data['x'])
g = nx.Graph()
g.add_node(N)

G = nx.complete_graph(N)
'''
nx.draw_networkx(G, pos = pos)



pts = mlab.points3d(data['x'], data['y'], data['z'],
                    scale_factor=0.1,
                    scale_mode='none',
                    colormap='Blues',
                    resolution=20)

pts.mlab_source.dataset.lines = np.array(list(G.edges()))

'''
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(data['x'], data['y'], data['z'])
'''
nx.draw_networkx(G, 
                 font_size=10,
                 alpha=.5,
                 width=.2,
                 with_labels=False,
                 pos=pos,
                 cmap=plt.cm.autumn)

'''
def draw_edges(graph, data):
    x = data['x'].values
    y = data['y'].values
    z = data['z'].values
    nodes = len(graph.nodes)

    for i in range(nodes):
        for k in range(5):
            j = np.random.randint(0, i+20)
            if j <=0 or j>=nodes:
                    j = i
            
            p_from = np.array([x[i], y[i], z[i]])
            p_to = np.array([x[j], y[j], z[j]])
            plt.plot([p_from[0], p_to[0]], [p_from[1], p_to[1]], [p_from[2], p_to[2]], 'k-', linewidth=0.3, alpha = 0.3)
            
   
draw_edges(G, data)
#%%
import pyqtgraph.opengl as gl


def load_nv(filename):
    number_of_vertices = np.loadtxt(filename, max_rows = 1).astype(int).item()
    vertices = np.loadtxt(filename, skiprows = 1, max_rows = number_of_vertices)
    number_of_faces = np.loadtxt(filename, skiprows = number_of_vertices+1, max_rows = 1).astype(int).item()
    faces = np.loadtxt(filename, skiprows = number_of_vertices+2, max_rows = number_of_faces).astype(int) -1
    return {'vertices':vertices, 'faces':faces}

brain_color = [0.7, 0.6, 0.55, 1.0]
mesh = load_nv('BrainMesh_ICBM152.nv')
colors = np.array([brain_color for i in range(len(mesh['faces']))])
brain_mesh = gl.GLMeshItem(vertexes=mesh['vertices'], faces=mesh['faces'], shader = 'normalColor')

test = gl.GLViewWidget()
test.addItem(brain_mesh)
test.show()



