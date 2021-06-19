# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 11:11:45 2021

@author: sagar kumar
"""

# HAMMING DISTANCE MEASURE
# Hamming distance calculates the distance between two binary vectors, 
# also referred to as binary strings or bitstrings for short.
#red = [1, 0, 0]
#green = [0, 1, 0]
#blue = [0, 0, 1]
#The distance between red and green could be calculated as the sum or the average number
# of bit differences between the two bitstrings. This is the Hamming distance.
#For a one-hot encoded string, it might make more sense to summarize to the sum of the
# bit differences between the strings, which will always be a 0 or 1


#calculating hamming distance
def hamming_distance(a,b):
    return sum(abs(a1-b1) for a1,b1 in zip(a,b))/len(a)

row1 = [0,0,0,1,0,0]
row2 = [0,0,0,0,1,0]
distance = hamming_distance(row1,row2)
print(distance)

#there are two differences between the strings, or 2 out of 6 bit positions different, 
#which averaged (2/6) is about 1/3 or 0.333.


# calculating hamming distance between bit strings
from scipy.spatial.distance import hamming
# define data
row1 = [0, 0, 0, 0, 0, 1]
row2 = [0, 0, 0, 0, 1, 0]
# calculate distance
dist = hamming(row1, row2)
print(dist)