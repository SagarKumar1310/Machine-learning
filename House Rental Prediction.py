#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


dataset = pd.read_csv('https://raw.githubusercontent.com/edyoda/data-science-complete-tutorial/master/Data/house_rental_data.csv.txt')


# In[3]:


dataset.head()


# In[4]:


df = dataset.set_index('Unnamed: 0')


# In[5]:


df.head()


# In[6]:


df.rename(columns={'Living.Room':'Livingroom'}, inplace=True)


# In[7]:


df.head()


# In[8]:


df.corr()


# In[17]:


df.info()


# In[19]:


df.isnull().sum()


# In[9]:


x = df.iloc[:,:-1].values


# In[10]:


y = df.iloc[:,6].values


# In[14]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)


# In[15]:


from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)


# In[16]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)
print(lr.score(x_test, y_test))


# In[ ]:




