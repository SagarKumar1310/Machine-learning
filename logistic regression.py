#!/usr/bin/env python
# coding: utf-8

# Exercise
# Download employee retention dataset from here: https://www.kaggle.com/giripujar/hr-analytics.
# 
# Now do some exploratory data analysis to figure out which variables have direct and clear impact on employee retention (i.e. whether they leave the company or continue to work)
# Plot bar charts showing impact of employee salaries on retention
# Plot bar charts showing corelation between department and employee retention
# Now build logistic regression model using variables that were narrowed down in step 1
# Measure the accuracy of the model

# In[3]:


import pandas as pd
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[20]:


df = pd.read_csv(r"C:\Users\sagar kumar\Downloads\HR_comma_sep.csv")
print(df.shape)
df.head()


# In[16]:


left = df[df.left==1]
left.shape


# In[17]:


retained = df[df.left==0]
retained.shape


# In[21]:


df.groupby('left').mean()


# In[22]:


pd.crosstab(df.salary,df.left).plot(kind='bar')


# In[23]:


pd.crosstab(df.Department,df.left).plot(kind='bar')


# In[24]:


subdf = df[['satisfaction_level','average_montly_hours','promotion_last_5years','salary']]
subdf.head()


# In[25]:


salary_dummies = pd.get_dummies(subdf.salary, prefix="salary")
df_with_dummies = pd.concat([subdf,salary_dummies],axis='columns')
df_with_dummies.head()


# In[28]:


df_with_dummies.drop('salary',axis='columns',inplace=True)
df_with_dummies.head()


# In[29]:


X = df_with_dummies
X.head()


# In[32]:


y = df.left
y.head()


# In[33]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.3)


# In[34]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()


# In[35]:


model.fit(X_train, y_train)


# In[36]:


model.predict(X_test)


# In[37]:


model.score(X_test,y_test)


# # Logistic Regression: Multiclass Classification

# In[38]:


from sklearn.datasets import load_digits
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
digits = load_digits()


# In[39]:



plt.gray() 
for i in range(5):
    plt.matshow(digits.images[i])


# In[40]:


dir(digits)


# In[41]:


digits.data[0]


# In[42]:



from sklearn.linear_model import LogisticRegression
model = LogisticRegression()


# In[43]:


from sklearn.model_selection import train_test_split


# In[44]:


X_train, X_test, y_train, y_test = train_test_split(digits.data,digits.target, test_size=0.2)


# In[45]:


model.fit(X_train, y_train)


# In[46]:


model.score(X_test, y_test)


# In[48]:


model.predict(digits.data[0:5])


# # Confusion Matrix

# In[49]:


y_predicted = model.predict(X_test)


# In[50]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predicted)
cm


# In[51]:


import seaborn as sn
plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')


# In[ ]:




