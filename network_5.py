########################
# Network to study pattern
# Learning with heterogenous connectivity
# Ramon Martinez 19 / 03 / 2014
########################

import numpy as np
import matplotlib.pyplot as plt
import h5py
from distance_functions import *
from time import localtime

def gradient(initial, end, N):
    '''
    Creates a gradient of values along the x dimension
    The N must be the size of the network
    '''
    aux = np.zeros((N,N))
    aux[:,:] = np.linspace(initial,end,N)
    return aux

def gausian(mu,sigma, N):
    '''
    Gausian distribution of network parameters
    '''
    return np.random.gausian(mu,sigma,(N,N))

def constant(constant, N):
    '''
    A simple matrix with constant values 
    '''
    matrix =  np.ones((N,N)) * constant
    return matrix

##########################
# Parameters 
#########################

# Neuron parameters 
E = 30
Vth = 0
Vre = -50
tau = 20

#Network parameters 
## Wave 
# N = 40
#alpha = 0
#beta = 0.10
#r_alpha = 10 # Excitation radio 
#r_beta = 15 # inhibition radio 

##Bumps
# N = 40
# alpha = constant(0, N)
# beta = constant(1, N)
# r_alpha = constant(6, N)
# r_beta = constant(30, N)

## gradient
N = 40
alpha = constant(0,N)
beta = gradient(0,1,N)
r_alpha = constant(6,N)
r_beta = constant(30, N)

# Time simulation parameters 
dt = 1.0
T = 500
Nt = int( T / dt)

# filename
directory = '../data/'
date_stamp = '%4d-%02d-%02dT%02d-%02d-%02d' % localtime()[:6]
format ='.hdf5'
title = 'experiment_hetero' 
filename = directory + date_stamp + title + format

##########################
# Simulation
##########################

## Initialize the network
# Homogeneus start 
#V = np.zeros([N,N])
#V[:] = Vre
# Random start 
V = np.random.rand(N,N) * (Vth - Vre) + Vre

## Save the files 
f = h5py.File(filename)
dset_voltage = f.create_dataset('voltage', shape=(N,N,Nt), dtype=np.float32)
dset_spikes = f.create_dataset('spikes', shape=(N,N,Nt),dtype=np.bool)

f.create_dataset('initial_state', data = V, dtype=np.float32)
f.create_dataset('network/alpha', data=alpha, dtype=np.float32)
f.create_dataset('network/beta', data=beta, dtype=np.float32)
f.create_dataset('network/r_alpha', data=r_alpha, dtype=np.float32)
f.create_dataset('network/r_beta', data=r_beta, dtype=np.float32)

# Store the neuron parameters 
dset_voltage.attrs['dt'] = dt
dset_voltage.attrs['E'] = E
dset_voltage.attrs['Vth'] = Vth
dset_voltage.attrs['Vre'] = Vre
dset_voltage.attrs['tau'] = tau


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
        #V += spatial_term_hom(V, N, index, alpha, r_alpha, beta, r_beta)
        V += spatial_term_het(V, N, index, alpha, r_alpha, beta, r_beta)

        # Substract the spikes so far from the new ones 
        new_spikes = ( V >  Vth) - spikes
        NAP = np.sum(new_spikes) #count the number of new spikes 
        spikes = V > Vth #Recalculate total spikes 

        
    # Reset the voltage
    V[spikes] = Vre
    
    # Save the data 
    dset_voltage[::,::,t] = V
    dset_spikes[::,::,t] = spikes  

#we also ask HDF5 to flush its buffers
#and actually write to disk
f.flush()


