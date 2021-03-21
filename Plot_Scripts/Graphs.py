


#%%

import math
import json
import numpy as np
import pandas as pd
import networkx as nx
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from IPython.display import Image
import json

names = ('airline,airline_id,'
         'source,source_id,'
         'dest,dest_id,'
         'codeshare,stops,equipment').split(',')
routes = pd.read_csv(
    'https://github.com/ipython-books/'
    'cookbook-2nd-data/blob/master/'
    'routes.dat?raw=true',
    names=names,
    header=None)

names = ('id,name,city,country,iata,icao,lat,lon,'
         'alt,timezone,dst,tz,type,source').split(',')
airports_all = pd.read_csv(
    'https://github.com/ipython-books/'
    'cookbook-2nd-data/blob/master/'
    'airports.dat?raw=true',
    header=None,
    names=names,
    index_col=4,
    na_values='\\N')


airports = airports_all
#airports = airports_all[airports_all['country'] ==
#                       'Italy']

routes = routes[
    routes['source'].isin(airports.index)  &
    routes['dest'].isin(airports.index)]


edges = routes[['source', 'dest']].values


g = nx.from_edgelist(edges)


fig, ax = plt.subplots(1, 1, figsize=(6, 6))
nx.draw_networkx(g, ax=ax, node_size=5,
                 font_size=6, alpha=.5,
                 width=.5)
ax.set_axis_off()

def connected_component_subgraphs(G):
    for c in nx.connected_components(G):
        yield G.subgraph(c)
        
        
sg = next(connected_component_subgraphs(g))
sg = nx.Graph(sg)
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
nx.draw_networkx(sg, ax=ax, with_labels=False,
                 node_size=5, width=.5)
ax.set_axis_off()


unique_index = set(airports.index)
unique_index.remove(np.nan)
airports_index = airports.index.isin(unique_index)
airports = airports[airports_index]


pos = {airport: (v['lon'], v['lat'])
       for airport, v in
       airports.to_dict('index').items()}


deg = nx.degree(sg)
sizes = [deg[iata] for iata in sg.nodes]


altitude = airports['alt']
altitude = [altitude[iata] for iata in sg.nodes]

labels = {iata: iata if deg[iata] >= 20 else ''
          for iata in sg.nodes}
#%%
# Map projection subplot_kw=dict(projection=crs)
crs = ccrs.PlateCarree()
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection = crs)
ax.stock_img()

# Extent of continental US.
#ax.set_extent([-20, 54, 70, 50])
base_uri = 'http://map1c.vis.earthdata.nasa.gov/wmts-geo/wmts.cgi'
layer_name = 'VIIRS_CityLights_2012'
ax.coastlines(color='')
nx.draw_networkx(sg, ax=ax,
                 font_size=10,
                 alpha=.5,
                 width=.2,
                 node_size=sizes,
                 with_labels=False,
                 pos=pos,
                 node_color=altitude,
                 cmap=plt.cm.autumn)


#%%
write = 'y'
if write == 'y':
    graph_json = nx.readwrite.json_graph.node_link_data(sg)
    with open('graph_data.json', 'w') as outfile:
        json.dump(graph_json, outfile)
    with  open('pos_dic.json', 'w') as outfile:
        json.dump(pos, outfile)







