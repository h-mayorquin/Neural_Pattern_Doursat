########################
# Network to study pattern learning
# Ramon Martinez 19 / 02 / 2014
########################

import numpy as np
import matplotlib.pyplot. as plt

E = 30
Vth = 0
Vre = -50
V0  = Vre
tau = 20

N = 2

dt = 0.1
T = 1
Nt = int( T / dt)

V = np.rand.rand() * (Vre - Vth) + Vre
print V
