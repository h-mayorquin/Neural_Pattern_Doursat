########################
# File to analyze the data produced by the networks  
# Ramon Martinez February / 2014
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

def visualize_network_V(V):
    """
    Shows a color map of the neuron voltages 
    """
    plt.imshow(V)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Voltage (mV)')
    plt.xlabel('Neuron\'s x coordinate')
    plt.ylabel('Neuron\'s y coordinate')
    plt.title('')
    #plt.clim(-50,10)
    plt.show()


def visualize_network_rate(rate):
    """
    Shows a color map of the neuron voltages 
    """
    plt.imshow(rate)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Firing Rate (Hz)')
    plt.xlabel('Neuron\'s x coordinate')
    plt.ylabel('Neuron\'s y coordinate')
    plt.title('Firing rate in a 20 ms window')
    plt.clim(0,70)
    plt.show()
    
    

########################
# Load the data 
########################
f = h5py.File('../data/experiment.hdf5')
# f = h5py.File('experiment')
voltage = f['voltage']
spikes = f['spikes']

dt = 0.1
N = np.shape(voltage)[0] # Number of neurons 
T = np.shape(voltage)[2] # Total time 

########################
# Calculate the mean rate 
########################

dm = 20 # Time window to the mean rate 

def calculate_mean_rate(spikes, N, T, dm, dt):

    T_window = T  - int(dm / dt)
    rate = np.zeros((N,N,T_window))

    for i in xrange(T_window):
        rate[:,:,i] = np.mean(spikes[:,:,i:(i+dm)],2)

    return rate * 1000 / dt #Transform to Hertz

rate = calculate_mean_rate(spikes, N, T, dm, dt)

########################
# Calculate the averages
########################
def calculate_average(quantity):
    return np.mean(quantity,2)

#Vavg = calculate_average(voltage)
#rate_avg = calculate_average(rate)


########################
# Animation
########################

def create_animation_voltage(voltage):
    #Initiate figure 
    fig = plt.figure()
    ims = plt.imshow(voltage[:,:,0])
    plt.xlabel('Neuron\'s x coordinate')
    plt.ylabel('Neuron\'s y coordinate')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Voltage (mV)')
    plt.clim(-50,0)

    # Define how wto update frames 
    def updatefig(i):
        ims.set_array( voltage[:,:,i] )
        return ims,
    # run and save the animation
    image_animation = animation.FuncAnimation(fig,updatefig, frames=20, interval=1, blit = True)
    image_animation.save('animationFunction_interval1_fps10_dpi=200.mp4', fps=10, dpi=200)


def create_animation_rate(rate):
    fig = plt.figure()
    ims = plt.imshow(rate[::,::,1])
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Firing Rate (Hz)')
    plt.xlabel('Neuron\'s x coordinate')
    plt.ylabel('Neuron\'s y coordinate')
    plt.title('Firing rate in a 20 ms window')
    plt.clim(0,70)
    
    # Define how wto update frames 
    def updatefig(i):
        ims.set_array( rate[:,:,i] )
        return ims,

    # run and save the animation
    image_animation = animation.FuncAnimation(fig,updatefig, frames=1000, interval=1, blit = True)
    image_animation.save('rate__frames=100-fps10_dpi=200.mp4', fps=10, dpi=200)



create_animation_rate(rate)
plt.close()
#plt.show()
