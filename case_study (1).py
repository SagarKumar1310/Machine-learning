#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd


# In[3]:


dataset = pd.read_csv('https://raw.githubusercontent.com/edyoda/data-science-complete-tutorial/master/Data/house_rental_data.csv.txt')


# In[4]:


dataset.head()


# In[5]:


dataset.tail()


# In[6]:


dataset.shape


# In[7]:


dataset.columns


# In[8]:


dataset.dtypes


# In[9]:


dataset.info()


# In[10]:


df = dataset.set_index("Unnamed: 0")


# In[11]:


df.rename(columns={'Living.Room':'Livingroom'}, inplace=True)


# In[12]:


df.head()


# In[13]:


df.describe(include='all')


# In[14]:


df.isnull().sum()


# In[15]:


df.corr()


# In[16]:


import seaborn as sns
sns.heatmap(df.corr())


# In[17]:


sns.jointplot(data=df , x = 'Sqft' , y = 'Price' , kind='scatter')


# In[18]:


sns.jointplot(data=df , x = 'Floor' , y = 'Price' , kind='scatter')


# In[19]:


sns.jointplot(data=df , x = 'Bedroom' , y = 'Price' , kind='scatter')


# In[20]:


sns.jointplot(data=df , x = 'Livingroom' , y = 'Price' , kind='scatter')


# In[21]:


sns.jointplot(data=df , x = 'Bathroom' , y = 'Price' , kind='scatter')


# In[22]:


sns.pairplot(data=df)


# In[23]:


df.head()


# In[24]:


x = df.iloc[:, 0:6].values
y = df.iloc[:, -1].values


# In[25]:


#Split into train and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.30, random_state = 0)


# In[26]:


#Perform standardization
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# In[27]:


#Apply KNN
from sklearn.neighbors import KNeighborsRegressor
nn_model = KNeighborsRegressor(n_neighbors=3)
nn_model.fit(x_train, y_train)
y_pred = nn_model.predict(x_test)


# In[28]:


#Chekc the train and test score
print(nn_model.score(x_train, y_train))
print(nn_model.score(x_test, y_test))


# In[29]:


#Chekc the ideal value of k
from sklearn.metrics import mean_squared_error
from math import sqrt
error = []

for k in range(2, 20):
    nn_model = KNeighborsRegressor(n_neighbors = k)
    nn_model.fit(x_train, y_train)
    y_predict = nn_model.predict(x_test)
    
    error_value = sqrt(mean_squared_error(y_test, y_predict))
    error.append(error_value)
    print(k, error_value)
    
graph = pd.DataFrame(error)
graph.plot()


# K = 5 

# In[ ]:




