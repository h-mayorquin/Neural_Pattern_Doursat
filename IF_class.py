import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

class Neuron:
    """
    Integrate and Fire Neuron Class

    Methods:
    constructor(V.tau,E,Vth,Vre,x,y): Sets the initial value of the parameters

    EvolveVoltage(dt): Given a value of dt uses the Euler Method
    to simple evolve the voltage

    Attribuites:
    V: Voltage of the neuron

    tau : Time constant of the neuron

    E : Driving field or current of the neuron. It is also the fix point of the
    EDO that controls the neuron

    Vth: Voltage Treshold, when the neuron arrives to this value is reseted to

    Vre: Reset voltage

    x,y: Position in space 
    """
    
    def __init__(self, V=-50, tau=20, E=10, Vth=0, Vre=-50, x=0 , y=0):
        self.V = V
        self.E = E
        self.tau = tau
        self.Vth = Vth
        self.Vre = Vre
        self.x = x
        self.y = y
        
    def EvolveVoltage(self, dt):
        self.V =  self.V + ( dt / self.tau ) * (self.E - self.V)
        if (self.V > self.Vth):
          self.V = self.Vre
        return self.V
        
