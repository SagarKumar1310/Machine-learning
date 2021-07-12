# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 14:38:42 2021

@author: sagar kumar
"""

import pandas as pd
dataset = pd.read_csv('train_set.csv')

YT = {'Fiscal': 0, 'Calendar':1}
dataset['YT'].replace(YT, inplace=True)

dtype = dataset.dtypes

df = dataset.select_dtypes(include=["int64","float64"])
df = df.drop("EI",axis = 1)
x = df.iloc[ : , :-1].values
y = df.iloc[ : ,-1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train, y_train)

print(model.score(x_train, y_train))
print(model.score(x_test, y_test))












