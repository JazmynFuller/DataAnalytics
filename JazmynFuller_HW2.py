#!/usr/bin/env python
# coding: utf-8

# ## HomeWork 2 EDA, Data Visualization and Linear Regression Model 

# # Due on 9/30/2019 mid-night

# ## Late Policy: Take off 50% after one day, 80% after two days

# In[154]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import random
import warnings
warnings.filterwarnings('ignore')


# ## Loading our MPG3 dataset

# In[155]:


mpg = pd.read_csv("mpg3.csv")
mpg.head()


# In[156]:


mpg.shape


# # Question 1: Check to see if there are any missing values. Fix the missing values by imputing value from the mean. After fixing missing values, you should still have 405 rows

# In[157]:


# Type your code here\
df = sns.load_dataset('mpg')
na_replacement = mpg.mean()
mpg.fillna(na_replacement, inplace=True)
mpg.isnull().any()
mpg.shape


# # Question 2 : Use Boxplot to see if there are any outliers on ALL of the numerical fields (Hint you should have 402 rows now after outliers removal)

# In[158]:


sns.boxplot(x=mpg['displacement']);


# In[159]:


sns.boxplot(x=mpg['acceleration']);


# In[160]:


sns.boxplot(x=mpg['model_year']);


# In[161]:


sns.boxplot(x=mpg['horsepower']);


# In[162]:


sns.boxplot(x=mpg['mpg']);


# In[163]:


sns.boxplot(x=mpg['weight']);


# In[164]:


sns.boxplot(x=mpg['cylinders']);


# ### Removing outliers

# In[165]:


mpg = mpg[mpg.mpg  < 90]
mpg = mpg[mpg.weight < 10000 ]
mpg = mpg[mpg.cylinders < 20 ]
mpg.shape


# # Question 3 : Remove any duplicates rows. Hint after removal, you should have 400 rows now

# In[166]:


# Show number of rows before
print("Before duplicates removal: ", mpg.shape)

# Type your code here to remove duplicated rows
mpg.drop_duplicates(keep='first',inplace=True)

# hint: Google how to remove duplicate rows in pandas dataframe and you will find the link
# https://pandas.pydata.org/pandas-docs/version/0.17/generated/pandas.DataFrame.drop_duplicates.html

# Show number of rows before
print("After duplicates removal: ", mpg.shape)


# # Question 4: Create a pair plot

# In[167]:


sns.pairplot(mpg);


# # Question 5: Create a FacetGrid of a scatter plot of mpg vs weight for different country

# In[168]:


g = sns.FacetGrid(mpg, col='origin')
g.map(plt.scatter, 'mpg', 'weight');


# # Now Load the adult income dataset and do some EDA and answer the following questions based on the Adult income dataset

# In[169]:


adult = pd.read_csv("adult.data.csv")
adult.head()


# In[170]:


adult.isnull().any()


# In[171]:


print(adult.shape)
adult.describe()


# # Question 6 : Which two martial-status is most common

# In[172]:


most_common_marital_status = adult['marital-status'].value_counts();
print(most_common_marital_status)
print("\nThe two most common marital-statuses are Married-civ-spuse (%s recorded) and Never-married (%s recorded)" % (most_common_marital_status.iloc[0], most_common_marital_status.iloc[1]))


# # Question 7: Plot the age distribution broken down by different martial-status using a FacetGrid

# In[173]:


h = sns.FacetGrid(adult, col='marital-status')
h.map(plt.hist,'age',bins=20);


# # Question 8: Create a Facet Grid for fnlwgt against age broken down by race and sex. You should have 10 sub-plots

# In[174]:


fnl_vs_age = adult[['fnlwgt','age','race','sex']]
m = sns.FacetGrid(fnl_vs_age, row='sex' ,col='race')
m = m.map(plt.hist,'age',bins=10).set_axis_labels("Age", "Fnlwgt")


# # Now Load the housing data and answer all the remaining questions based on the housing dataset

# In[175]:


housing = pd.read_csv("USA_housing.csv")
housing.head()


# In[176]:


# a scatter plot comparing num_children and num_pets
housing.plot(kind='scatter',x='Income',y='Price',color='blue');
plt.show()
housing.shape


# In[177]:


housing.plot(kind='scatter', x = "HouseAge", y = "Price", color = "blue");


# In[178]:


housing.plot(kind='scatter', x = "NumberOfRooms", y = "Price", color = "green");


# # Question 9: Use a pair plot or individual scatter plots, pick ONE variable that best explains house price

# In[179]:


# Type your answer (ie which variable affect the Price the most)
housing.plot(kind='scatter', x = "HouseAge", y = "Price", color = "blue");
housing.plot(kind='scatter', x = "NumberOfRooms", y = "Price", color = "orange");
housing.plot(kind='scatter', x = "Population", y = "Price", color = "red");
housing.plot(kind='scatter', x = "AreaNumberOfBedrooms", y = "Price", color = "green");
housing.plot(kind='scatter', x = "Income", y = "Price", color = "purple");

print('\nBased on the plots, I believe the income of the homeowners best explains house price.')
#housing.plot(kind='scatter', x = "AreaNumberOfBedrooms", y = "Price", color = "yellow");


# In[180]:


# from sklearn.model_selection import train_test_split 
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
# model2 = LinearRegression()
# model2.fit(X_train, Y_train)
# print('R-squared:', metrics.r2_score(Y_test, Y_pred))


# # Question 10:  Use that variable to build a one-variable Linear Regression model of the house price. Make sure you split the data between Training and Testing set first, Save 20% as your testing data

# In[181]:


Xarray = housing['Income'].values
Yarray = housing['Price'].values
X = Xarray.reshape(-1, 1)
Y = Yarray.reshape(-1, 1)
model = LinearRegression()
model.fit(X, Y)


# ## Split data values

# In[182]:


# Show your R-square and RMSE
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
model = LinearRegression()
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)
print('R-squared: ', metrics.r2_score(Y_test, Y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(Y_test, Y_pred)))


# ## Non-Split

# In[183]:


modelA = LinearRegression()
modelA.fit(X, Y)
Y_predA = modelA.predict(X)
print('R-squared: ', metrics.r2_score(Y, Y_predA))
print('RMSE:', np.sqrt(metrics.mean_squared_error(Y, Y_predA)))


# In[185]:


plt.scatter(X_test, Y_test,  color='gray')
plt.scatter(X_test, Y_pred, color='red', linewidth=2)
plt.show()


# # Question 11: Now do a 10-fold Cross Validation of your model. Does your model pass cross validation?

# In[186]:


# Show your work here
rsquare = []
for i in range(10):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state = random.randint(1,1000))
    model1 = LinearRegression()
    model1.fit(X_train, Y_train)
    Y_pred = model1.predict(X_test)
    rsquare.append(metrics.r2_score(Y_test, Y_pred))
    
r2df = pd.DataFrame({'Trial':range(10), 'R_squared': rsquare})
r2df

r2df.plot.bar(x='Trial',y='R_squared')
x = r2df['R_squared'].mean();
print('The mean of the r-squared values is %s \n' % x)
print('My model appears to pass the testing data as the R-squared values are within the same range')


# # Question 12: Now pick ONE more variable in addition to the variable you choose from Question 9 and build a two-variables Linear Regression model of the house price. Make sure you split the data into training and testing set first

# In[187]:


# Show your work here
X = housing[['Population','Income']].values.reshape(-4, 2)
Y = housing['Price'].values.reshape(-1, 1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0);


# # Question 13: Do a 10-fold Cross Validation of your two-variable model. Does your model pass cross validation?

# In[188]:


# Show your work here
rsquare = []
for i in range(10):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state = random.randint(1,1000))
    model3 = LinearRegression()
    model3.fit(X_train, Y_train)
    Y_pred = model3.predict(X_test)
    rsquare.append(metrics.r2_score(Y_test, Y_pred))
    
r2df1 = pd.DataFrame( {'trial': range(10), 'Rsquare': rsquare})
r2df1.plot.bar(x='trial', y='Rsquare')
print('The mean of the r-squared values is %s' % r2df1['Rsquare'].mean())
print('Based off of my r-squared value, it passes the cross validation.')


# In[ ]:





# In[ ]:




