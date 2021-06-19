# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 19:54:58 2021

@author: sagar kumar
"""

from sklearn.datasets import load_boston
import pandas as pd
boston = load_boston()

#values in data frame
df = pd.DataFrame(boston.data)
df.columns = boston.feature_names
df["price"] = boston.target

#split into x and y
X = df.iloc[:,:-1 ].values
y = df.iloc[:, -1].values

#split in test and train
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)

#perform standarization
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)

#apply KNN
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
nn_model = KNeighborsRegressor(n_neighbors=5)
y_pred = nn_model.predict(X_test)
nn_model.fit()

