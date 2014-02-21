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

def calculate_index(aux2):
      index = []
      for k in range(N):
          index.append( [aux2[0][k],aux2[1][k]] )

      return 0

def distance(i,j,index):
    neuron_position = np.array([i,j])
    squares = (neuron_position - index) ** 2
    distance = np.sqrt(np.sum(squares, 1))
    return distance

def distance2(i, j, aux2, dimension):
    neuron_position = np.array([i,j])
    distances = np.zeros(len(aux2[0]))

    for k,(a,b) in enumerate(zip(aux2[0],aux2[1])):
        distances[k] = circular_distance(neuron_position, [a,b],dimension)

    return distances



def circular_distance(p1, p2, dimension):        
    total = 0
    for (x, y) in zip(p1, p2):
        delta = abs(x - y)
        if delta > dimension - delta:
            delta = dimension - delta
        total += delta ** 2
    return total ** 0.5

##########################
# Parameters 
#########################

# Neuron parameters 
E = 10
Vth = 0
Vre = -30
V0  = Vre
tau = 20

# Network parameters 
N = 30
alpha = 2
beta = 1
r_alpha = 3
r_beta = 15

# Time simulation parameters 
dt = 0.1
T = 500
Nt = int( T / dt)

##########################
# Simulation
##########################


# Initialize the network
V = np.zeros([N,N])
V[:] = Vre
V = np.random.rand(N,N) * (Vth - Vre) + Vre

Vavg = np.zeros([N,N])
Vavg2 = np.zeros([N,N])

# Evolve the network 
for t in range(Nt):
     # Evolve the voltage 
    V = V + ( dt / tau ) * (E - V)
    n = t + 1
    Vavg = ( 1.0 / n ) * ( (n - 1) * Vavg + V )
    # Register action potentials 
    AP = V > Vth
    NAP = np.sum(AP) # Number of actions potentials 
    if (NAP > 0):
        # Store a list with the indexes of spiking neurons
        index = np.where(AP)
                  
        for i in range(N):
            for j in range(N):
                # Calculate distance between each neuron and
                # the neurons that have spiked 
                dis = distance2(i,j,index,N)
                
                # For each neuron that spike add the excitatory
                # and inhibitory effect to the neuron voltage 
                excitation = alpha * np.sum( dis  < r_alpha)
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
    
visualize_network(Vavg)  

