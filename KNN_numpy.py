# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 15:21:17 2021

@author: sagar kumar
"""

#import datasets
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
iris = load_iris()
df = pd.DataFrame(iris.data,columns = iris.feature_names)
df["type"] = iris.target

class KNN:

#select value of K
   def __init__(self,k):
       self.k = k
       
   def my_fit(self,feature_data,target_data):
       self.feature_data = np.array(feature_data)
       self.target_data = np.array(target_data)


#calculate Euclidean distance
   def calculate_Euclidean_distance(self,one_data):
       distance = np.sqrt(np.sum(np.square(self.feature_data - one_data),axis = 1))
       return distance

#find K neighbor
   def find_K_neighbor(self,one_data):
       res = self.calculate_Euclidean_distance(one_data)
       return res.argsort()[:self.k]

#find K Neighbor class
   def find_K_neighbor_class(self,one_data):
       index_of_neighbor = self.find_K_neighbor(one_data)
       return self.target_data[index_of_neighbor]
   
   def my_pred(self,one_data):
       classes = self.find_K_neighbor_class(one_data)
       return np.bincount(classes).argmax()
   
model = KNN(5)
feature_data = df.drop(columns=['type'], axis = 1)
target_data = df.type
model.my_fit(feature_data, target_data)

one_data = [1,2,3,4]
#model.find_k_neighbors_class(one_data)
print(model.my_pred(one_data))




#my predict