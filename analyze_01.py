########################
# File to analyze the data produced by the networks  
#
########################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import h5py


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


########################
# Load the data 
########################
f = h5py.File('../data/experiment.hdf5')
# f = h5py.File('experiment')
voltage = f['voltage']
spikes = f['spikes']

N = np.shape(voltage)[0] # Number of neurons 
T = np.shape(voltage)[2] # Total time 

########################
# Calculate the mean rate 
########################

dm = 20 # Time window to the mean rate 
T_window = T  - dm*10
rate = np.zeros((N,N,T_window))

for i in range(T_window):
    rate[:,:,i] = np.mean(spikes[:,:,i:(i+dm)],2)
    
########################
# Calculate the averages
########################

Vavg = np.mean(voltage,3)
rate_avg =np.mean(rate,3)


