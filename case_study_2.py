#!/usr/bin/env python
# coding: utf-8

# In[106]:


import pandas as pd
dataset = pd.read_csv("https://raw.githubusercontent.com/edyoda/data-science-complete-tutorial/master/Data/house_rental_data.csv.txt")
dataset.head()


# In[107]:


df = dataset.set_index("Unnamed: 0")
df.head()


# In[108]:


df.tail()


# In[109]:


df.shape


# In[110]:


dataset.columns


# In[111]:


dataset.dtypes


# In[112]:


dataset.info()


# In[113]:


df.rename(columns={'Living.Room':'Livingroom'}, inplace=True)
df.head()


# In[114]:


df.describe(include='all')


# In[115]:


df.isnull().sum()


# In[116]:


df.corr()


# In[117]:


x = df.iloc[ : ,0:6].values
x


# In[161]:


from sklearn.cluster import KMeans
k_mean = KMeans(n_clusters=5,init ="k-means++",random_state=4)
k_mean.fit(x)
print(k_mean.labels_)


# In[152]:


wcss=[]
for k in range(1,15):
    k_mean = KMeans(n_clusters=k,init ="k-means++",random_state=4)
    k_mean.fit(x)
    wcss.append(k_mean.inertia_)
    
    
import matplotlib.pyplot as plt
plt.plot(range(1,15),wcss)
plt.title("Elbow_method")
plt.xlabel("no. of cluster")
plt.ylabel("wcss score")
plt.show()


# K = 3

# In[153]:


from sklearn.neighbors import NearestNeighbors


# In[154]:


nn = NearestNeighbors(n_neighbors=3)


# In[155]:


from sklearn.preprocessing import StandardScaler


# In[156]:


scaler = StandardScaler()


# In[157]:


X_tf = scaler.fit_transform(x)


# In[158]:


nn.fit(X_tf)


# In[159]:


data = nn.kneighbors(X_tf[:1])[1][0]


# In[160]:


df.iloc[data]


# In[ ]:





# In[ ]:




