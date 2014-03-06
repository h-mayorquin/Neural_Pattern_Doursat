########################
# File to analyze the data produced by the networks  
# Ramon Martinez February / 2014
########################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from visualization_functions import *
from time import localtime
import h5py


# Analysis functions 
def calculate_mean_rate(spikes, N, T, dm, dt):

    T_window = T  - int(dm / dt)
    rate = np.zeros((N,N,T_window))

    for i in xrange(T_window):
        rate[:,:,i] = np.mean(spikes[:,:,i:(i+dm)],2)

    return rate * 1000 / dt #Transform to Hertz

def calculate_average(quantity):
    return np.mean(quantity,2)    


########################
# Load the data 
########################
# Data to analyze
file = '2014-03-06T16-59-38wave'
format = '.hdf5'
file = file + format
f = h5py.File('../data/'+file)
# f = h5py.File('experiment')
voltage = f['voltage']
spikes = f['spikes']

dt = 1.0
N = np.shape(voltage)[0] # Number of neurons 
T = np.shape(voltage)[2] # Total time 

########################
# Calculate the mean rate 
########################

dm = 20 # Time window to the mean rate 
rate = calculate_mean_rate(spikes, N, T, dm, dt)

########################
# Calculate the averages
########################

Vavg = calculate_average(voltage)
rate_avg = calculate_average(rate)

########################
# Visualize
########################

# Animation parameters 
interval = 1 #Draws a new animation every interval millisecond
frames = int(T / dt) # Shows as many frames as data points 
fps = int(1 / dt) # One data per second  -multiply this if desired-
dpi = 90 # Quality 

# Where to save the file 
directory = '../data/'
date_stamp = '%4d-%02d-%02dT%02d-%02d-%02d' % localtime()[:6]
filename = directory + date_stamp 

#create_animation_voltage(voltage,frames,interval,fps,dpi,filename)
#create_animation_rate(rate,frames - int(dm/dt),interval,fps,dpi,filename)
visualize_network_V(Vavg)
visualize_network_rate(rate_avg)
#visualize_network_both(Vavg, rate_avg)

