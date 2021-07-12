# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 13:44:23 2021

@author: sagar kumar
"""

import pandas as pd
dataset = pd.read_csv("https://raw.githubusercontent.com/tranghth-lux/data-science-complete-tutorial/master/Data/HR_comma_sep.csv.txt")
dataset.rename(columns={'sales':'dept'}, inplace=True)
df = pd.get_dummies(dataset)
column = df.pop("left")
df.insert(20,"left",column)

x = df.iloc[:, :-1].values
y = df.iloc[:,-1].values

del df['salary_medium']
del df['dept_technical']

corr = df.corr()

#Splitting data into train and test set
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(classifier.score(x_test, y_test))


