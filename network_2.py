########################
# Network to study pattern learning
# Ramon Martinez 19 / 02 / 2014
########################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

def Visualize_distr(V):
    Vdis = V.reshape(len(V)*len(V))
    plt.plot(Vdis,'*')
    plt.show()

def Visualize_network(V):
    plt.imshow(V,interpolation = 'Nearest')
    plt.colorbar()
    plt.show()


##########################
# Parameters 
#########################

# Neuron parameters 
E = 50
Vth = 0
Vre = -20
V0  = Vre
tau = 20

# Network parameters 
N = 2

# Time simulation parameters 
dt = 0.1
T = 100
Nt = int( T / dt)
print Nt

##########################
# Simulation
##########################


# Initialize the network
V = np.random.rand(N,N) * (Vth - Vre) + Vre

# Evolve the network 
for i in range(Nt):
     # Evolve the voltage 
    V = V + ( dt / tau ) * (E - V)
    # Register action potentials 
    AP = V > Vth
    NAP = np.sum(AP) # Number of actions potentials 
    if (NAP > 0):
        # Store a list with the indexes of spiking neurons
        aux2 = np.where(AP)
        index = []
        for k in range(NAP):
            index.append( [aux2[0][k],aux2[1][k]] )
            
        print 'Action potential time', i * dt
        print 'index', index
        print '-------------'

    # Reset the voltage
    V[ AP ] = Vre
    
    
# Comentario adheredido    
# Segundo comeentario adherido 
