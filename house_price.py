# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 20:35:33 2021

@author: sagar kumar
"""

import pandas as pd
dataset = pd.read_csv('train.csv')
corr = dataset.corr()

del dataset['Id']
del dataset['Alley']
del dataset['PoolQC']
del dataset['Fence']
del dataset['MiscFeature']
dataset = dataset.dropna(axis = 0)


df = pd.get_dummies(dataset)
x = df.drop(['SalePrice'],1).values
y = df.SalePrice.values



from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)
print(lr.score(x_test, y_test))