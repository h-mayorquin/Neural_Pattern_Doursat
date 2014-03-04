########################
# Network to study pattern learning
# Ramon Martinez 19 / 02 / 2014
########################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import csv
import h5py


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

def spatial_term(V, N, index, alpha, r_alpha, beta, r_beta):
    """
    This function alculates the spatial effects to be added
    the voltage 
    """
     # Here we calculate the spatial effects 
            for i in xrange(N):
                for j in xrange(N):
                    # Calculate distance between each neuron and
                    # the neurons that have spiked 
                    dis = distance(i,j,index,N)
                
                    # For each neuron that spike add the excitatory
                    # and inhibitory effect to the neuron voltage 
                    excitation = alpha * np.sum( dis  < r_alpha)
                    inhibition = beta * np.sum( ( dis < r_beta) & ( r_alpha < dis ))
                V[i][j] += excitation - inhibition
            return V
       

def save_text(V,f):
    """
    Save text as a file 
    """
    V[ spikes ] = Vre
    for i in xrange(N):
        for j in xrange(N):
            f.write(str(V[i][j])+'_')
        f.write('\n') #Jump at the end of the column

def save_text_csv(V,mywriter):
    for row in V:
        mywriter.writerow(row)
        mywriter.writerow([])
   
  

##########################
# Parameters 
#########################

# Neuron parameters 
E = 10
Vth = 0
Vre = -50
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

## Initialize the network
# Homogeneus start 
#V = np.zeros([N,N])
#V[:] = Vre
# Random start 
V = np.random.rand(N,N) * (Vth - Vre) + Vre

# This will meassure the average voltage 
Vavg = np.zeros([N,N])

## Creates the files to save 
f = h5py.File('../data/experiment.hdf5')
dset_voltage = f.create_dataset('voltage', shape=(N,N,Nt), dtype=np.float32)
dset_spikes = f.create_dataset('spikes', shape=(N,N,Nt),dtype=np.bool)

# Evolve the network 
for t in xrange(Nt):
    # Evolve the voltage 
    V = V + ( dt / tau ) * (E - V)
    # Register action potentials 
    spikes = V > Vth
    NAP = np.sum(spikes) # Number of actions potentials 

    new_spikes = spikes
    while (NAP > 0):
        # Store a list with the indexes of the spiking neurons
        index = np.where(new_spikes)
        # Calculate the spatial effect 
        V += spatial_term(V, N, index, alpha, r_alpha, beta, r_beta)

        # Substract the spikes so far from the new ones 
        new_spikes = ( V >  Vth) - spikes
        NAP = np.sum(new_spikes) #count the number of new spikes 
        spikes = V > Vth #Recalculate total spikes 

        
    # Reset the voltage
    V[spikes] = Vre
    
    # Save the data 
    dset_voltage[::,::,t] = V
    dset_spikes[::,::,t] = spikes  

f.flush()


