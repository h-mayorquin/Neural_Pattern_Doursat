import numpy as np

def circular_distance(p1, p2, dimension):        
    """
    Euclidean distance with periodic boundary conditions
    In particular this measures the distance in a a grid
    with dimension as the number of squres 
    """
    total = 0
    for (x, y) in zip(p1, p2):
        delta = abs(x - y)
        if delta > dimension - delta:
            delta = dimension - delta
        total += delta ** 2
    return total ** 0.5

def distance(i, j, index, dimension):
    """
    Given a particular localation an array of other locations
    it returns an array with the distances from the individual
    neurons to every element in index
    """
    neuron_position = np.array([i,j])
    distances = np.zeros(len(index[0]))

    for k,(a,b) in enumerate(zip(index[0],index[1])):
        distances[k] = circular_distance(neuron_position, [a,b],dimension)

    return distances

def spatial_term_hom(V, N, index, alpha, r_alpha, beta, r_beta):
    """
    This function alculates the spatial effects to be added
    the voltage 
    """
    spatial_term = np.zeros((N,N))
    
     # Here we calculate the spatial effects 
    for i in xrange(N):
        for j in xrange(N):
            # Calculate distance between each neuron and
            # the neurons that have spiked 
            dis = distance(i,j,index,N)
            
            # For each neuron that spike add the excitatory
            # and inhibitory effect to the neuron voltage 
            excitation = alpha * np.sum( dis  < r_alpha)
            inhibition = beta * np.sum( ( dis < r_beta) & ( r_alpha < dis ))

            spatial_term[i][j] += excitation - inhibition

    return spatial_term
       

def spatial_term_het(V, N , index, alpha, r_alpha, beta, r_beta):
    """
    This is the function that calculates the spatial contribuition for heteregenous
    spatial effects. This method requieres alpha, r_alpha, beta and r_beta to be
    matrices of teh size of V (that is, of the size of the neuron lattice) in order to
    work properly.

    For homogenous case use spatial_term_hom
    """
    spatial_term = np.zeros((N,N))
    
    for i in xrange(N):
        for j in xrange(N):
            # Neuron of interest 
            neuron_index = [i,j]
            excitation = 0
            inhibition = 0
            for k in xrange(len(index[0])):
                # Calculate the coordinates of the spiking neuron
                spike_index = (index[0][k], index[1][k])
                # Calculate the distance between the spike
                # and the neuron of interest 
                distance = circular_distance(neuron_index, spike_index, N)
                # calculate the positive contribution
                excitation += alpha[spike_index] * (distance < r_alpha[spike_index] )
                # calculate the negative contribution
                inhibition += beta[spike_index] * ( ( r_alpha[spike_index] < distance) & ( distance < r_beta[spike_index] ) )


        spatial_term[i][j] = excitation - inhibition

    return spatial_term
