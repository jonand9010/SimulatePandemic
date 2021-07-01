import numpy as np

class NetworkSimulation:
    def __init__(self, Network, timesteps):
        self.Number_of_nodes = Network.Number_of_nodes
        self.nodes = list(Network.Graph.nodes())
        self.node_population = np.zeros((Network.Number_of_nodes, timesteps), dtype = 'int')
        self.A = Network.A
        self.A_I = np.zeros(Network.A.shape)  #Initializing adjacency matrix for infected
        self.L_sym = Network.L_sym


class SIR_NetworkSimulation(NetworkSimulation):

    def __init__(self, Network, timesteps):
        super().__init__(Network, timesteps)
        self.alpha, self.beta, self.gamma = Network.alpha, Network.beta, Network.gamma
        self.timesteps = timesteps
        self.S, self.I, self.R = self.node_population.copy(), self.node_population.copy(), self.node_population.copy()      #Initialize vector for susceptible, infected, and recovered in each city
        self.SIR = np.zeros((3, self.timesteps))
        self.start_num_infected = 10
        
        self.node_population[:,0] = 10000 * np.ones((self.node_population.shape[0], )) #Population in each city at t=0
        
        self.citykey = 'WUH'
        
        self.init_first_infected_city()

    def init_first_infected_city(self):

        for i in range(self.Number_of_nodes):

            if self.nodes[i] == self.citykey: #Selecting Wuhan as first infected city
                start_city_index = i

        self.I[start_city_index,0] = np.random.randint(self.start_num_infected)          # Number of infected in start city 
        self.S[:,0] = self.node_population[:,0] - self.I[:,0]                            # Defining the number of susceptible in each city at t=0
        self.SIR = np.zeros(shape = (3, self.timesteps))                                 # Initialize total SIR matrix
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


    def update_infected_adjacency_matrix(self, n, Flux_I):


        while Flux_I > 0:
            edge_list = np.where(self.A[n,:] == 1)[0]     #List of non-zero edges
            random_edge = np.random.choice(edge_list)   #Random choice of edge
            Nr_I  = np.random.randint(0, Flux_I+1) #Number of random infected to move in a certain direction
            self.A_I[random_edge, n] = Nr_I  #Nr_I infected moving along edge random_edge 
            self.A_I[n, n] = -Nr_I
            Flux_I = Flux_I - Nr_I #Count down the total number of infected left to distribute

    def simulate(self):
        for t in range(self.timesteps-1):
            dNdt = - self.alpha * np.dot(self.L_sym, self.node_population[:, t]) # Number of people travelling to another city each day

            dNdt = np.round(dNdt)

            print(f"Timestep: {t+1} of { self.timesteps-1}")
    
            for n in range(self.Number_of_nodes):

                Flux_I = self.get_infected_travellers(self.Number_of_nodes, self.node_population[n, t], self.I[n, t], dNdt[n])
                
                self.update_infected_adjacency_matrix(n, Flux_I)
            
            #Correction from movements
            dI = np.sum(self.A_I, axis = 1)

            self.I[:, t] = self.I[:, t] + dI 
            self.S[:, t] = self.S[:, t] - dI
            
            dSdt = -self.beta * self.I[:, t] * self.S[:, t] / self.node_population[:, t]
            dIdt = self.beta * self.I[:, t] * self.S[:, t] / self.node_population[:, t] - self.gamma * self.I[:, t]
            dRdt = self.gamma*self.I[:, t]
            
            self.S[:, t+1] = self.S[:, t] + dSdt
            self.I[:, t+1] = self.I[:, t] + dIdt
            self.R[:, t+1] = self.R[:, t] + dRdt
            self.node_population[:, t+1] = self.S[:, t+1] + self.I[:, t+1] + self.R[:, t+1]

            self.SIR[:, t+1] = np.sum(self.S[:,t+1]), np.sum(self.I[:,t+1]), np.sum(self.R[:,t+1])

        self.node_population[:, self.timesteps-1] = self.S[:, self.timesteps-1] + self.I[:, self.timesteps-1] + self.R[:, self.timesteps-1]
