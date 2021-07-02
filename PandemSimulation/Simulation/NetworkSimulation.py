import numpy as np

class NetworkSimulation:
    def __init__(self, Network, timesteps):
        self.Number_of_nodes = Network.Number_of_nodes
        self.nodes = list(Network.Graph.nodes())
        self.node_population = np.zeros((Network.Number_of_nodes, timesteps), dtype = 'int')
        self.A = Network.A
        self.node_degree = np.sum(self.A,0)
        self.A_I = np.zeros(Network.A.shape)  #Initializing adjacency matrix for infected
        self.L_sym = Network.L_sym
        self.travel_matrix = self.get_travel_matrix()



class SIR_NetworkSimulation(NetworkSimulation):

    def __init__(self, Network, timesteps):
        super().__init__(Network, timesteps)
        self.alpha, self.beta, self.gamma = Network.alpha, Network.beta, Network.gamma
        self.travel_parameter = Network.travel_parameter
        self.timesteps = timesteps
        self.S, self.I, self.R = self.node_population.copy(), self.node_population.copy(), self.node_population.copy()      #Initialize vector for susceptible, infected, and recovered in each city
        self.SIR = np.zeros((3, self.timesteps))
        self.start_num_infected = 10
        
        self.node_population[:,0] = 10000 * np.ones((self.node_population.shape[0], )) #Population in each city at t=0
        self.travel_matrix = self.get_travel_matrix()

        self.citykey = 'WUH'
        
        self.get_first_infected_city()

    def get_travel_matrix(self, travel_rate = 0.01):
    
        travel_matrix = self.A.copy()

        for i in range(self.Number_of_nodes):
            travel_matrix[i,:] = travel_rate * self.node_population[i, 0] / self.node_degree[i] * self.A[i, :]

        travel_matrix = np.floor(travel_matrix)

        return travel_matrix

    def get_first_infected_city(self):

        for i in range(self.Number_of_nodes):

            if self.nodes[i] == self.citykey: #Selecting Wuhan as first infected city
                start_city_index = i

        self.I[start_city_index,0] = np.random.randint(self.start_num_infected)          # Number of infected in start city 
        self.S[:,0] = self.node_population[:,0] - self.I[:,0]                            # Defining the number of susceptible in each city at t=0
        self.SIR = np.zeros(shape = (3, self.timesteps))                                 # Initialize total SIR matrix
        self.SIR[:, 0] = np.sum(self.S[:,0]), np.sum(self.I[:,0]), np.sum(self.R[:,0])



    def simulate(self, infection_rate = 1):
        for t in range(self.timesteps-1):

            nodal_infection_ratio = self.I[:, t]/self.node_population[:, t]

            dIdt = infection_rate* np.dot(self.travel_matrix, nodal_infection_ratio)
            dIdt = np.floor(dIdt)

            print(f"Timestep: {t+1} of { self.timesteps-1}")

            self.I[:, t] = self.I[:, t] + dIdt 
            self.S[:, t] = self.S[:, t] - dIdt
            
            dSdt = -self.beta * self.I[:, t] * self.S[:, t] / self.node_population[:, t]
            dIdt = self.beta * self.I[:, t] * self.S[:, t] / self.node_population[:, t] - self.gamma * self.I[:, t]
            dRdt = self.gamma*self.I[:, t]
            
            self.S[:, t+1] = self.S[:, t] + dSdt
            self.I[:, t+1] = self.I[:, t] + dIdt
            self.R[:, t+1] = self.R[:, t] + dRdt
            self.node_population[:, t+1] = self.S[:, t+1] + self.I[:, t+1] + self.R[:, t+1]

            self.SIR[:, t+1] = np.sum(self.S[:,t+1]), np.sum(self.I[:,t+1]), np.sum(self.R[:,t+1])

        self.node_population[:, self.timesteps-1] = self.S[:, self.timesteps-1] + self.I[:, self.timesteps-1] + self.R[:, self.timesteps-1]
