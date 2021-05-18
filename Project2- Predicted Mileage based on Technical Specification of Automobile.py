#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries and Dataset

# In[59]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

get_ipython().run_line_magic('matplotlib', 'inline')


# In[41]:


automob= pd.read_csv(r"C:\Users\shruti\Desktop\Decodr Session Recording\Project\Decodr Project\Predict Mileage based on Technical Specifications of Automobile\auto-mpg.csv")


# In[42]:


automob.head()


# In[43]:


automob.tail()


# ### Data Processing

# In[44]:


# Removing unnecessary columns

automob.drop(["car name"], axis=1, inplace= True)


# In[45]:


automob.head()


# In[46]:


automob.describe()


# In[47]:


# Checking for Null Values

automob.isnull().sum()


# In[48]:


# Checking for False Values

automob["horsepower"].unique()


# In[49]:


automob= automob[automob.horsepower!= "?"]


# In[50]:


"?" in automob


# In[51]:


automob.shape


# ### Correlation Matrix

# In[52]:


automob.corr()["mpg"].sort_values()


# In[53]:


# Plotting Heatmap of Correlation

plt.figure(figsize=(10,10))
sns.heatmap(automob.corr(), annot=True, linewidths= 0.5, center= 0, cmap= "rainbow")
plt.show()


# ### Univariate Analysis

# In[60]:


sns.countplot(automob.cylinders, data= automob, palette="rainbow")
plt.show()


# In[61]:


sns.countplot(automob["model year"], palette= "rainbow")
plt.show()


# In[62]:


sns.countplot(automob["origin"], palette= "rainbow")
plt.show()


# ### Multivariate Analysis

# In[63]:


sns.boxplot(x= "cylinders", y= "mpg", data= automob, palette= "rainbow")
plt.show()


# In[64]:


sns.boxplot(x= "model year", y= "mpg", data= automob, palette= "rainbow")
plt.show()


# In[79]:


# Modelling Dataset

x=automob.iloc[:,1:].values
y=automob.iloc[:,0].values


# ### Train and test data split

# In[97]:


from sklearn.model_selection import train_test_split


# In[98]:


x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=0.3, random_state=0)


# ### Building the Model

# In[99]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# In[100]:


regression= LinearRegression()
regression.fit(x_train, y_train)


# In[90]:


y_pred= regression.predict(x_test)


# In[92]:


print(regression.score(x_test, y_test))


# ### Polynomial Regression

# In[104]:


from sklearn.preprocessing import PolynomialFeatures
poly_reg= PolynomialFeatures(degree=2)


# In[105]:


x_poly= poly_reg.fit_transform(x)


# In[107]:


x_train, x_test, y_train, y_test= train_test_split(x_poly,y,test_size=0.3, random_state=0)

lin_regression= LinearRegression()
lin_regression.fit(x_train, y_train)

print(lin_regression.score(x_test, y_test))


# ### Conclusion

# In[ ]:


Accuracy score improves in Polynomial Regression compared to Linear Regression because it fits data much better.

