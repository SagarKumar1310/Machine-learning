# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 20:06:12 2021

@author: sagar kumar
"""
import pandas as pd
dataset = pd.read_csv('Nutrition__Physical_Activity__and_Obesity_-_Behavioral_Risk_Factor_Surveillance_System.csv')
df = dataset.drop(['YearStart', 'YearEnd', 'LocationDesc', 'Datasource',
        'Topic', 'Question', 'Data_Value_Unit', 'Data_Value_Type',
       'Data_Value', 'Data_Value_Alt', 'Data_Value_Footnote_Symbol',
       'Data_Value_Footnote',
       'Sample_Size', 'Total',
       'Race/Ethnicity', 'GeoLocation', 'ClassID', 'TopicID', 'QuestionID',
       'DataValueTypeID', 'LocationID', 'StratificationCategory1',
       'Stratification1', 'StratificationCategoryId1', 'StratificationID1'],1)
df = pd.get_dummies(df)
df = df.dropna(axis = 0)
corr = df.corr()
X = df.iloc[:, 0:79].values

from sklearn.cluster import KMeans
k_means = KMeans(n_clusters = 6, init = 'k-means++', random_state = 4)
k_means.fit(X)
print(k_means.labels_)


wcss = []
for k in range(1, 15):
    k_means = KMeans(n_clusters = k, init = 'k-means++', random_state = 4)
    k_means.fit(X)
    wcss.append(k_means.inertia_)
    


