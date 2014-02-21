########################
# Program to test circular distance 
#  Ramon Martinez 19 / 02 / 2014 
########################

import numpy as np

def circular_distance(p1, p2, dimension):        
    total = 0
    for (x, y) in zip(p1, p2):
        delta = abs(x - y)
        if delta > dimension - delta:
            delta = dimension - delta
        total += delta ** 2
    return total ** 0.5

N = 4
print 'Espace of four units '

p1 = np.array([0,2])
p2 = np.array([4,2])
result = circular_distance(p1,p2,N)
print 'The distance between (0,2) and (4,2) should be 0 = ',result 

p1 = np.array([0,0])
p2 = np.array([3,3])

result = circular_distance(p1,p2,N)
print 'The distance betweeen (0,0) and (3,3) be 1.4 = ',result


p1 = np.array([0,3])
p2 = np.array([0,0])
result = circular_distance(p1,p2,N)
print 'the distance between (0,0) and (0,3) should be 1 =', result 



