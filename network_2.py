########################
# Network to study pattern learning
# Ramon Martinez 19 / 02 / 2014
########################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

def visualize_distr(V):
    Vdist = V.reshape(len(V)*len(V))
    plt.plot(Vdist,'*')
    plt.show()

def visualize_network(V):
    plt.imshow(V,interpolation = 'Nearest')
    plt.colorbar()
    plt.show()

def calculate_index():
    
def distance(i,j,index):
     neuron_position = np.array([i,j])
     squares = (neuron_position - index) ** 2
     distance = np.sqrt(np.sum(squares, 1))
     return distance 


##########################
# Parameters 
#########################

# Neuron parameters 
E = 50
Vth = 0
Vre = -20
V0  = Vre
tau = 20

# Network parameters 
N = 30
alpha = 2
beta = 1
r_alpha = 1
r_beta = 3 

# Time simulation parameters 
dt = 0.1
T = 1000
Nt = int( T / dt)
print Nt

##########################
# Simulation
##########################


# Initialize the network
V = np.random.rand(N,N) * (Vth - Vre) + Vre
#V = np.zeros([N,N])
#V[:] = Vre

# Evolve the network 
for t in range(Nt):
     # Evolve the voltage 
    V = V + ( dt / tau ) * (E - V)
    # Register action potentials 
    AP = V > Vth
    NAP = np.sum(AP) # Number of actions potentials 
    if (NAP > 0):
        # Store a list with the indexes of spiking neurons
        aux2 = np.where(AP)
        index = []
        index2 = np.zeros([NAP,2])
               
        for k in range(NAP):
            index.append( [aux2[0][k],aux2[1][k]] )
            index2[k][0] = aux2[0][k]
            index2[k][1] = aux2[1][k]


        for i in range(N):
            for j in range(N):
                # Calculate distance between each neuron and
                # the neurons that have spiked 
                dis = distance(i,j,index)

                # For each neuron that spike add the excitatory
                # and inhibitory effect to the neuron voltage 
                excitation = alpha * np.sum( dis  < r_alpha))
                inhibition = beta * np.sum( ( dis < r_beta) & ( r_alpha < dis ))
                V[i][j] += excitation - inhibition

        print 'PRINT TO DEBUG'
        print 'Action potential time', t * dt
        print 'where', aux2
        print 'index', index
        print 'index2', index2
        print '-------------'

    # Reset the voltage
    V[ AP ] = Vre
    
visualize_network(V)  
