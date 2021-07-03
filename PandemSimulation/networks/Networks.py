import numpy as np
import networkx as nx
import json

class SIRNetwork:

    def __init__(self, datafiles, travel_rate, beta, gamma, travel_infection_rate = 1):
        self.graph, self.A, self.pos = self.load_graph(datafiles['data'], datafiles['position'])
        self.nodes = list(self.graph.nodes())
        self.node_degree = np.sum(self.A, 0)
        self.number_of_nodes = self.A.shape[0]
        self.travel_rate = travel_rate
        self.beta = beta
        self.gamma = gamma
        self.R0 = self.beta/self.gamma
        self.travel_infection_rate = travel_infection_rate


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
