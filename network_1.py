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

E = 30
Vth = 0
Vre = -50
V0  = Vre
tau = 20

N = 2

dt = 0.1
T = 0.4
Nt = int( T / dt)
print Nt

V = np.random.rand(N,N) * (Vth - Vre) + Vre
#Visualize_distr(V)

fig, ax = plt.subplots()

for i in range(Nt):
    # Evolve the voltage 
    V = V + ( dt / tau ) * (E - V)
    # Restet the voltage 
    V[ V > Vth ] = Vre
    if i==0:
        points, = ax.imshow(V)
    else:
        new_x = ax.imshow(V)
        points.set_data(new_x)
    plt.pause(0.5)

# This didn't work 

