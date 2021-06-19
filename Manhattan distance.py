# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 11:32:16 2021

@author: sagar kumar
"""

#Manhattan Distance
#The Manhattan distance, also called the Taxicab distance or the City Block distance, 
#calculates the distance between two real-valued vectors.

#It is perhaps more useful to vectors that describe objects on a uniform grid, 
#like a chessboard or city blocks. The taxicab name for the measure refers to the 
#intuition for what the measure calculates: the shortest path that a taxicab would 
#take between city blocks (coordinates on the grid).

#sum of all the distance


# calculating manhattan distance between vectors
from math import sqrt

# calculate manhattan distance
def manhattan_distance(a, b):
	return sum(abs(e1-e2) for e1, e2 in zip(a,b))

# define data
row1 = [10, 20, 15, 10, 5]
row2 = [12, 24, 18, 8, 7]
# calculate distance
dist = manhattan_distance(row1, row2)
print(dist)


# calculating manhattan distance between vectors
from scipy.spatial.distance import cityblock
# define data
row1 = [10, 20, 15, 10, 5]
row2 = [12, 24, 18, 8, 7]
# calculate distance
dist = cityblock(row1, row2)
print(dist)
