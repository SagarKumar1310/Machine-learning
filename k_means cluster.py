# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 10:16:59 2021

@author: sagar kumar
"""

import pandas as pd
dataset = pd.read_csv("https://raw.githubusercontent.com/edyoda/DS281220-ML/main/Mall_Customers.csv")
x = dataset.iloc[ : ,2:5].values

from sklearn.cluster import KMeans
k_mean = KMeans(n_clusters=5,init ="k-means++",random_state=4)
k_mean.fit(x)
print(k_mean.labels_)

wcss=[]
for k in range(1,15):
    k_mean = KMeans(n_clusters=k,init ="k-means++",random_state=4)
    k_mean.fit(x)
    wcss.append(k_mean.inertia_)
    
import matplotlib.pyplot as plt
plt.plot(range(1,15),wcss)
plt.title("Elbow_method")
plt.xlabel("no. of cluster")
plt.ylabel("wcss score")
plt.show()
    