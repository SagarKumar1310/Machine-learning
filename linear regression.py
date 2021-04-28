#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
from word2number import w2n


# In[86]:


df = pd.read_csv("https://raw.githubusercontent.com/codebasics/py/master/ML/1_linear_reg/Exercise/canada_per_capita_income.csv")
df.head()


# In[48]:


df.rename({'per capita income (US$)':'income'},axis=1,inplace=True)
df.head(1)


# In[49]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.xlabel('year')
plt.ylabel('income')
plt.scatter(df.year,df.income ,color='red',marker='+')


# In[82]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.xlabel('year',fontsize = 20)
plt.ylabel('income',fontsize = 20)
plt.scatter(df.year,df.income ,color='red',marker='+')
plt.plot(df.year,reg.predict(df[['year']]),color='blue')


# In[72]:


new_df = df.drop('income',axis='columns')
new_df.head()


# In[71]:


income = df.income
income.head()


# In[73]:


# Create linear regression object
reg = linear_model.LinearRegression()
reg.fit(new_df,income)


# In[74]:


reg.predict([[2020]])


# ## 2nd question

# In[21]:


import pandas as pd
import numpy as np
from sklearn import linear_model
from word2number import w2n


# In[22]:


df = pd.read_csv("https://raw.githubusercontent.com/codebasics/py/master/ML/2_linear_reg_multivariate/Exercise/hiring.csv")
df


# In[23]:


df.experience = df.experience.fillna("zero")
df


# In[24]:


df.experience = df.experience.apply(w2n.word_to_num)
df


# In[25]:


import math
median_test_score = math.floor(df['test_score(out of 10)'].mean())
median_test_score


# In[26]:


df['test_score(out of 10)'] = df['test_score(out of 10)'].fillna(median_test_score)
df


# In[27]:


reg = linear_model.LinearRegression()
reg.fit(df[['experience','test_score(out of 10)','interview_score(out of 10)']],df['salary($)'])


# In[28]:


reg.predict([[2,9,6]])


# In[29]:


reg.predict([[12,10,10]])


# ## cost function

# In[30]:


import pandas as pd


# In[31]:


df = pd.read_csv("https://raw.githubusercontent.com/codebasics/py/master/ML/3_gradient_descent/Exercise/test_scores.csv")
df


# In[73]:


import numpy as np

def gradient_descent(x,y):
    m_curr = b_curr = 0
    iterations = 10
    n = len(x)
    learning_rate = 0.00001

    for i in range(iterations):
        y_predicted = m_curr * x + b_curr
        cost = (1/n) * sum([val**2 for val in (y-y_predicted)])
        md = -(2/n)*sum(x*(y-y_predicted))
        bd = -(2/n)*sum(y-y_predicted)
        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd
        print ("m {}, b {}, cost {} iteration {}".format(m_curr,b_curr,cost, i))

x = np.array(df.cs)
y = np.array(df.math)

gradient_descent(x,y)


# #  one hot coding by pandas

# In[75]:


import pandas as pd


# In[76]:


df = pd.read_csv("https://raw.githubusercontent.com/codebasics/py/master/ML/5_one_hot_encoding/Exercise/carprices.csv")
df


# In[91]:


dummies=pd.get_dummies(df['Car Model'])
dummies


# In[98]:


merge = pd.concat([df,dummies],axis = 'columns')
merge


# In[102]:


final = merge.drop(['Car Model','Mercedez Benz C class'],axis = 'columns')
final


# In[104]:


x = final.drop(['Sell Price($)'],axis= 'columns')
x


# In[111]:


y = final['Sell Price($)']
y


# In[112]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()


# In[115]:


model.fit(x,y)


# In[116]:


model.score(x,y)


# In[118]:


model.predict([[45000,4,0,0]])


# In[117]:


model.predict([[86000,7,0,1]])


# #  one hot coding by sklearn

# In[147]:


df


# In[148]:


from sklearn.preprocessing import LabelEncoder 
le = LabelEncoder()


# In[149]:


dfle = df
dfle['Car Model'] = le.fit_transform(dfle['Car Model'])
dfle


# In[150]:


x = dfle[['Mileage','Age(yrs)']].values
x


# In[151]:


y = dfle['Sell Price($)'].values
y


# In[152]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('Car Model', OneHotEncoder(), [0])], remainder = 'passthrough')


# In[153]:


x = ct.fit_transform(x)
x


# In[154]:


x = x[:,1:]
x


# In[155]:


model.fit(x,y)


# In[ ]:




