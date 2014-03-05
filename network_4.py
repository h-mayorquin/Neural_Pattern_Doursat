########################
# Network to study pattern learning
# Ramon Martinez 19 / 02 / 2014
########################

import numpy as np
import matplotlib.pyplot as plt
import h5py
from distance_functions import *

def gradient(initial, end, size):
    aux = np.zeros((size,size))
    aux[:,:] = np.linspace(initial,end,size)
    return aux

def gausian(mu,sigma,N):
    return np.random.gausian(mu,sigma,(N,N))


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
N = 10
alpha = 2
beta = 1


beta = gradient(0,1,N)
r_alpha = 3
r_beta = 15

# Time simulation parameters 
dt = 0.1
T = 10
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

## Creates the files to save 
f = h5py.File('../data/hetero_data.hdf5')
dset_voltage = f.create_dataset('voltage', shape=(N,N,Nt), dtype=np.float32)
dset_spikes = f.create_dataset('spikes', shape=(N,N,Nt),dtype=np.bool)
dset_spatial
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
        V += spatial_term_hom(V, N, index, alpha, r_alpha, beta, r_beta)
        #V += spatial_term_het(V, N, index, alpha, r_alpha, beta, r_beta)

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


