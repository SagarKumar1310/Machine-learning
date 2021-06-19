# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 12:56:34 2021

@author: sagar kumar
"""

import pandas as pd
datasets = pd.read_csv("https://raw.githubusercontent.com/edyoda/DS281220-ML/main/Social_Network_Ads.csv")
x = datasets.iloc[:,[2,3]].values
y = datasets.iloc[:,4].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(x_train,y_train)

y_pred = classifier.predict(x_test)

from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))

import numpy as np
error_rate = []
for i in range(2, 25):
    classifier = KNeighborsClassifier(n_neighbors=i)
    classifier.fit(x_train, y_train)
    i_pred = classifier.predict(x_test)
    error_rate.append(np.mean(i_pred != y_test))
    
print(error_rate)