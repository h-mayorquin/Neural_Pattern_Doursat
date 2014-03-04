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


def visualize_network_rate(V):
    """
    Shows a color map of the neuron voltages 
    """
    plt.imshow(V)
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

def calculate_mean_rate(spikes,N,T,dm,dt):

    T_window = T  - int(dm / dt)
    rate = np.zeros((N,N,T_window))

    for i in xrange(T_window):
        rate[:,:,i] = np.mean(spikes[:,:,i:(i+dm)],2)

    return rate * 1000 / dt #Transform to Hertz

#rate = calculate_mean_rate(spikes, N, T, dm, dt)

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

fig = plt.figure()

#ims = []
#ims.append((plt.pcolor(x, y, base + add, norm=plt.Normalize(0, 30)),))
#for i in range(100):
#    aux = plt.imshow(voltage[:,:,i],interpolation='None')
#    ims.append( (aux, ) )

#image_animation = animation.ArtistAnimation(fig, ims, interval=1)
#image_animation.save('animation.mp4')

ims = plt.imshow(voltage[:,:,0])
plt.xlabel('Neuron\'s x coordinate')
plt.ylabel('Neuron\'s y coordinate')
cbar = plt.colorbar()
cbar.ax.set_ylabel('Voltage (mV)')
plt.clim(-50,0)

def updatefig(i):
    ims.set_array( voltage[:,:,i] )
    return ims,

image_animation = animation.FuncAnimation(fig,updatefig,frames=500, interval=5, blit = True)

image_animation.save('animation_interval5_fps1_dpi=200.mp4',fps=1)

plt.close()
#plt.show()
