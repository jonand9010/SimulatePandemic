import numpy as np
import networkx as nx
import json

class SIRNetwork:

    def __init__(self, datafiles, alpha, beta, gamma):
        self.Graph, self.A, self.pos = self.load_graph(datafiles['data'], datafiles['position'])
        self.Number_of_nodes = self.A.shape[0]
        self.L_sym = self.Laplacian(self.A)    # Calculate symmetric graph Laplacian
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