#######################################################
# Program to test the neuon class
# Ramon Heberto Martinez
# 12 / 02 / 2014 
#######################################################

import numpy as np
import matplotlib.pyplot as plt
import IF_class

#####
# Initialize neuron
#####

I = 30
Vth = 0
Vre = -50
V0  = Vre
tau = 20

neuron = IF_class.Neuron(V = V0, E = I, tau = tau, Vth = Vth, Vre = Vre,x = 0, y = 0)

#####
# Simulation
#####

T = 100 # Total time of the simulation
dt = 0.1 # Time step
N = int(T / dt) # Number of steps that the simulations takes 
voltage = np.zeros(N)
time = np.zeros(N)

#Evolve the neuron 
for i in range(N):
    voltage[i] = neuron.EvolveVoltage(dt)
    time[i] = i
    
####
# Ploting 
####

plt.plot(time,voltage)
plt.show()

