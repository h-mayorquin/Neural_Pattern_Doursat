#######################################################
# Program to test the reticulate and design its class 
# Ramon Heberto Martinez
# 12 / 02 / 2014 
#######################################################

import numpy as np 
import matplotlib.pyplot as plt
from IF_class import Neuron

##
# Parameters 
##

# Grid parameters 
N = 2 # Number of neurons 
d = 1 # Distance between the neurons 

#Neuron parameters 
V0 = 0
E = 10
tau = 20
Vth = 0
Vre = 0

#Time simulation parameters
dt = 0.1
T = 10
Nt = int(T / dt)

##
# Simulation
## 

#Intialize the grid 
grid = []
for i in range(N):
    for j in range(N):
        grid.append(Neuron(x=i,y=j))

#Evolve the grid
for t in range(Nt):
    for i in range(N):
        for j in range(N):
            grid[i][j].EvolveVoltage(dt)
        

# Visualize the grid         
