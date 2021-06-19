# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 11:36:15 2021

@author: sagar kumar
"""

#Minkowski Distance
#Minkowski distance calculates the distance between two real-valued vectors.
#It is a generalization of the Euclidean and Manhattan distance measures and adds a 
#parameter, called the “order” or “p“, that allows different distance measures to be calculated.

#p=1: Manhattan distance.
#p=2: Euclidean distance.


# calculating minkowski distance between vectors
from math import sqrt

# calculate minkowski distance
def minkowski_distance(a, b, p):
	return sum(abs(e1-e2)**p for e1, e2 in zip(a,b))**(1/p)

# define data
row1 = [10, 20, 15, 10, 5]
row2 = [12, 24, 18, 8, 7]
# calculate distance (p=1)
dist = minkowski_distance(row1, row2, 1)
print(dist)
# calculate distance (p=2)
dist = minkowski_distance(row1, row2, 2)
print(dist)


# calculating minkowski distance between vectors
from scipy.spatial import minkowski_distance
# define data
row1 = [10, 20, 15, 10, 5]
row2 = [12, 24, 18, 8, 7]
# calculate distance (p=1)
dist = minkowski_distance(row1, row2, 1)
print(dist)
# calculate distance (p=2)
dist = minkowski_distance(row1, row2, 2)
print(dist)
