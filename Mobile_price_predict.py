# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 10:53:53 2021

@author: sagar kumar
"""

import pandas as pd
dataset = pd.read_csv('mobile_price_data.csv')
dataset['mobile_price'] = dataset['mobile_price'].str.replace(r'\D','')

dataset['mobile_price']=dataset['mobile_price'].astype(int)
datatype = dataset.dtypes

del dataset['mobile_name']
del dataset['dual_sim']
del dataset['bluetooth']
del dataset['mobile_color']



x = dataset.drop('mobile_price',axis = 1)
y = dataset.mobile_price


x = pd.get_dummies(x)

x = x.values
y = y.values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.30, random_state =0 )



from sklearn.neighbors import KNeighborsRegressor
nn_model = KNeighborsRegressor(n_neighbors=5)
nn_model.fit(x_train, y_train)
y_pred = nn_model.predict(x_test)


#Chekc the train and test score
print(nn_model.score(x_train, y_train))
print(nn_model.score(x_test, y_test))