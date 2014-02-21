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
alpha = 2
beta = 1
r_alpha = 1
r_beta = 3 

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
#V = np.zeros([N,N])
#V[:] = Vre

# Evolve the network 
for t in range(Nt):
     # Evolve the voltage 
    V = V + ( dt / tau ) * (E - V)
    # Register action potentials 
    AP = V > Vth
    NAP = np.sum(AP) # Number of actions potentials 
    if (NAP > 0):
        # Store a list with the indexes of spiking neurons
        aux2 = np.where(AP)
        index = []
        index2 = np.zeros([NAP,2])
               
        for k in range(NAP):
            index.append( [aux2[0][k],aux2[1][k]] )
            index2[k][0] = aux2[0][k]
            index2[k][1] = aux2[1][k]


        for i in range(N):
            for j in range(N):
                a = np.array([i,j])
                b = (a - index) ** 2
                e = np.sqrt(np.sum(b, 1))
                excitation = alpha * np.sum(( e < r_alpha))
                inhibition = beta  * np.sum( ( e < r_beta) & ( r_alpha < e ))
                V[i][j] += excitation - inhibition
                
        print 'Action potential time', t * dt
        print 'where', aux2
        print 'index', index
        print 'index2', index2
        print '-------------'

    # Reset the voltage
    V[ AP ] = Vre
    
Visualize_network(V)  
