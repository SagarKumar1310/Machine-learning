# -*- coding: utf-8 -*-
"""
Created on Sun Aug 15 13:47:28 2021

@author: sagar kumar
"""

import pandas as pd
dataset = pd.read_csv("https://raw.githubusercontent.com/edyoda/data-science-complete-tutorial/master/Data/epilepsy.data",delimiter=",")
df = dataset.drop("name",axis = 1)
df = df.dropna()
x = df.drop("status",axis = 1)
y = df.status
corr = x.corr()
del x["MDVP:Jitter(%)"]
del x["MDVP:Jitter(Abs)"]
del x["MDVP:APQ"]
del x["MDVP:RAP"]


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))
print(classifier.score(x_test,y_test))

