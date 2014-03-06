####################################################
# Visualization functions 
###############################################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


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
    Shows a color map of the neuron rates 
    """
    plt.imshow(rate)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Firing Rate (Hz)')
    plt.xlabel('Neuron\'s x coordinate')
    plt.ylabel('Neuron\'s y coordinate')
    plt.title('Firing rate in a 20 ms window')
    plt.clim(0,70)
    plt.show()
    
def create_animation_voltage(voltage, frames=100, interval=1, fps=10, dpi=90, filename='default', format ='.mp4'):
    """
    Documentation needed 
    """
    
    filename = filename + '-voltage_animation' + format
    #Initiate figure 
    fig = plt.figure()
    ims = plt.imshow(voltage[:,:,0])
    plt.xlabel('Neuron\'s x coordinate')
    plt.ylabel('Neuron\'s y coordinate')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Voltage (mV)')
    plt.clim(-50,0)

    # Define how to update frames 
    def updatefig(i):
        ims.set_array( voltage[:,:,i] )
        return ims,
    # run and save the animation
    image_animation = animation.FuncAnimation(fig,updatefig, frames=frames, interval=interval, blit = True)
    image_animation.save(filename, fps=fps, dpi=dpi)
    plt.show()

def create_animation_rate(rate, frames=100, interval=1, fps=10, dpi=90, filename='default.mp4', format='.mp4'):
    """
    Documentation needed 
    """
    
    filename = filename + '-rate_animation' + format
    # Initiate figure 
    fig = plt.figure()
    ims = plt.imshow(rate[::,::,1])
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Firing Rate (Hz)')
    plt.xlabel('Neuron\'s x coordinate')
    plt.ylabel('Neuron\'s y coordinate')
    plt.title('Firing rate in a 20 ms window')
    plt.clim(0,70)
    
    # Define how to update frames 
    def updatefig(i):
        ims.set_array( rate[:,:,i] )
        return ims,

    # run and save the animation
    image_animation = animation.FuncAnimation(fig,updatefig, frames=frames, interval=interval, blit = True)
    image_animation.save(filename, fps=fps, dpi=dpi)
    plt.show()

