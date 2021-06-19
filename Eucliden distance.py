# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 11:26:19 2021

@author: sagar kumar
"""

#Euclidean Distance
#Euclidean distance calculates the distance between two real-valued vectors.
#If columns have values with differing scales, it is common to normalize or standardize 
#the numerical values across all columns prior to calculating the Euclidean distance. 
#Otherwise, columns that have large values will dominate the distance measure.

#formula is same as pythogrous a^2=b^2-c^2


# calculating euclidean distance between vectors
from math import sqrt
 
# calculate euclidean distance
def euclidean_distance(a, b):
	return sqrt(sum((e1-e2)**2 for e1, e2 in zip(a,b)))
 
# define data
row1 = [10, 20, 15, 10, 5]
row2 = [12, 24, 18, 8, 7]
# calculate distance
dist = euclidean_distance(row1, row2)
print(dist)


# calculating euclidean distance between vectors
from scipy.spatial.distance import euclidean
# define data
row1 = [10, 20, 15, 10, 5]
row2 = [12, 24, 18, 8, 7]
# calculate distance
dist = euclidean(row1, row2)
print(dist)