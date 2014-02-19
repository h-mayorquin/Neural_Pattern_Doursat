##################################
# Program to construct an IF-network using
# Brian
# Ramon Martinez 19 / 02 / 2014
##################################

from brian import *

# Parameters 
tau = 20 * msecond        # membrane time constant
Vt = -50 * mvolt          # spike threshold
Vr = -60 * mvolt          # reset value
El = -49 * mvolt          # resting potential (same as the reset)
psp = 0.5 * mvolt         # postsynaptic potential size

# Define the network
G = NeuronGroup(N=40, model='dV/dt = -(V-El)/tau : volt',
              threshold=Vt, reset=Vr)
# Define the connections 
C = Connection(G,G,sparseness=0.1,weight=psp)

# Set the spike monitor
M = SpikeMonitor(G)
M2 = StateMonitor(G,'V', record=True)

# Intialize the values 
G.V = Vr + rand(40) * (Vt - Vr)

# Run the simulation
run(0.2 * second)

# Plot and outpu
plot(M2.times / ms, M2[0] / mV)
xlabel('Time (in ms)')
ylabel('Membrane potential (in mV)')
title('Membrane potential for neuron 0')
show()
