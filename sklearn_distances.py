# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 12:22:08 2021

@author: sagar kumar
"""

from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances, cosine_distances

x = [[0,1],[1,1]]
y = [[1,2],[2,2]]

print(euclidean_distances(x,y))
print(manhattan_distances(x,y))
print(cosine_distances(x,y))