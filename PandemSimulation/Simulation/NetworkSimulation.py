import numpy as np

class SIR_NetworkSimulation:

    def __init__(self, Network, timesteps):

        self.timesteps = timesteps
        self.N = np.zeros((Network.Number_of_nodes, timesteps), dtype = 'int')   #Initialize vector for city populations
        self.S, self.I, self.R = self.N.copy(), self.N.copy(), self.N.copy()      #Initialize vector for susceptible, infected, and recovered in each city
        self.A_I = np.zeros((Network.Number_of_nodes, Network.Number_of_nodes))  #Initializing adjacency matrix for infected
        self.SIR = np.zeros((3, self.timesteps))
        
        self.N[:,0] = 10000 * np.ones((self.N.shape[0], )) #Population in each city at t=0
        
        list_pos = list(Network.pos.keys())
        pos_list = list(Network.pos.items())

        for i in range(Network.Number_of_nodes):

            if list(Network.Graph.nodes())[i] == 'WUH': #Selecting Wuhan as first infected city
                start_pos = i

        r_node = start_pos
        self.I[r_node,0] = np.random.randint(10)          #Random number of infected in city r_node
        self.S[:,0] = self.N[:,0] - self.I[:,0]        # Defining the number of susceptible in each city at t=0



        self.SIR = np.zeros(shape = (3, self.timesteps))  #Initialize total SIR matrix
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
        for t in range(self.timesteps-1):
            dNdt = - Network.alpha * np.dot(Network.L_sym, self.N[:, t]) # Number of people travelling to another city each day

            dNdt = np.round(dNdt)

            print(f"Timestep: {t+1} of { self.timesteps-1}")
    
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

        self.N[:, self.timesteps-1] = self.S[:, self.timesteps-1] + self.I[:, self.timesteps-1] + self.R[:, self.timesteps-1]
