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
E = 30
Vth = 0
Vre = -50
tau = 20

#Network parameters 
# Wave 
N = 40
alpha = 0
beta = 0.10
r_alpha = 10
r_beta = 14

# Bumps
# N = 40
# alpha = 0
# beta = 1
# r_alpha = 6
# r_beta = 30


# Time simulation parameters 
dt = 0.1
T = 250
Nt = int( T / dt)

# filename
directory = '../data/'
date_stamp = '%4d-%02d-%02dT%02d-%02d-%02d' % localtime()[:6]
title = 'wave'
filename = directory + date_stamp + title

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
f = h5py.File(filename)
dset_voltage = f.create_dataset('voltage', shape=(N,N,Nt), dtype=np.float32)
dset_spikes = f.create_dataset('spikes', shape=(N,N,Nt),dtype=np.bool)


# Evolve the network 
for t in xrange(Nt):
    print t*dt
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


