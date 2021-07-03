import matplotlib.pyplot as plt
import networkx as nx
import cartopy.crs as ccrs
from matplotlib.animation import FuncAnimation
import numpy as np




class WorldMapSimulation:

    def __init__(self, Network, Simulation):
        self.fig = plt.figure()
        self.ax = plt.subplot(111, projection = ccrs.PlateCarree())
        self.timesteps = Simulation.timesteps
        self.Network = Network
        self.Simulation = Simulation

    def run_simulation(self):        
        
        for t in range(self.timesteps):

            self.ax.stock_img()
            self.ax.coastlines()

            nx.draw_networkx(self.Network.Graph, ax = self.ax, font_size=10,
                    alpha=.25,
                    width=.1,
                    node_size=0.1*self.Simulation.I[:,t],
                    with_labels=False,
                    pos=self.Network.pos,
                    node_color = 'r',
                    cmap=plt.cm.autumn)

            plt.pause(0.05)
            
            if t != self.timesteps-1:
                if self.Simulation.SIR[1, t] == 0:
                    break
                
                self.ax.cla()
        
class SIRHistoryPlot():

    def __init__(self, Simulation):
        
        self.fig = plt.figure()
        self.ax = plt.subplot(111)
        self.l1, = self.ax.plot(Simulation.SIR[0]/Simulation.SIR[0,0], 'b', alpha = 0.5)
        self.l2, = self.ax.plot(Simulation.SIR[1]/Simulation.SIR[0,0], 'r', alpha = 0.5)
        self.l3, = self.ax.plot(Simulation.SIR[2]/Simulation.SIR[0,0], 'g', alpha = 0.5)
        self.ax.legend([self.l1, self.l2, self.l3], ['Susceptible', 'Infected', 'Recovered'])
        self.ax.set_xlabel('Days')
        plt.show()
    
    def save_fig(path = './'):
        plt.savefig(path + 'Epidemic' + str(t) + '.png')
