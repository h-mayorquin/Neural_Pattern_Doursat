########################
# Network to study pattern learning
# Ramon Martinez 19 / 02 / 2014
########################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

def visualize_distr(V):
    """
    Shows the distribuition of voltages
    """
    Vdist = V.reshape(len(V)*len(V))
    plt.plot(Vdist,'*')
    plt.show()

def visualize_network(V):
    """
    Shows a color map of the neuron voltages 
    """
    plt.imshow(V,interpolation = 'Nearest')
    plt.colorbar()
    plt.show()

def distance(i, j, index, dimension):
    """
    Given a particular localation an array of other locations
    it returns an array with the distances from the individual
    neurons to every element in aux 
    """
    neuron_position = np.array([i,j])
    distances = np.zeros(len(index[0]))

    for k,(a,b) in enumerate(zip(index[0],index[1])):
        distances[k] = circular_distance(neuron_position, [a,b],dimension)

    return distances

def circular_distance(p1, p2, dimension):        
    """
    Euclidean distance with periodic boundary conditions
    In particular this measures the distance in a a grid
    with dimension as the number of squres 
    """
    total = 0
    for (x, y) in zip(p1, p2):
        delta = abs(x - y)
        if delta > dimension - delta:
            delta = dimension - delta
        total += delta ** 2
    return total ** 0.5

def save_text(V,f):
    """
    Save text as a file 
    """
    V[ spikes ] = Vre
    for i in range(N):
        for j in range(N):
            f.write(str(V[i][j])+'_')
        f.write('\n')

  

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
N = 10 
alpha = 2
beta = 1
r_alpha = 3
r_beta = 15

# Time simulation parameters 
dt = 0.1
T = 50
Nt = int( T / dt)

##########################
# Simulation
##########################

# Initialize the network
# Homogeneus start 
#V = np.zeros([N,N])
#V[:] = Vre
# Random start 
V = np.random.rand(N,N) * (Vth - Vre) + Vre

# This will meassure the average voltage 
Vavg = np.zeros([N,N])

f = open('../data/try.txt','w+')


# Evolve the network 
for t in range(Nt):
    # Evolve the voltage 
    V = V + ( dt / tau ) * (E - V)
    # Here we calculate the average 
    n = t + 1
    Vavg = ( 1.0 / n ) * ( (n - 1) * Vavg + V )

    # Register action potentials 
    spikes = V > Vth
    NAP = np.sum(spikes) # Number of actions potentials 

    new_spikes = spikes
    while (NAP > 0):
        # Store a list with the indexes of the spiking neurons
        index = np.where(new_spikes)

        # Here we calculate the spatial effects 
        for i in range(N):
            for j in range(N):
                # Calculate distance between each neuron and
                # the neurons that have spiked 
                dis = distance(i,j,index,N)
                
                # For each neuron that spike add the excitatory
                # and inhibitory effect to the neuron voltage 
                excitation = alpha * np.sum( dis  < r_alpha)
                inhibition = beta * np.sum( ( dis < r_beta) & ( r_alpha < dis ))
                V[i][j] += excitation - inhibition

        # Substract the spikes so far from the new ones 
        new_spikes = ( V >  Vth) - spikes
        NAP = np.sum(new_spikes) #count the number of new spikes 
        spikes = V > Vth #Recalculate total spikes 

        
    # Reset the voltage
    save_text(V,f)

f.close()

