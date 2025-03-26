#!/usr/bin/env python
# coding: utf-8

# ## London Housing Data Analysis Report

# ### Introduction
# 
# This report presents an Exploratory Data Analysis (EDA) of the London housing dataset. The analysis aims to uncover insights into housing prices, the impact of various features, and highlight key trends across different time periods and property types. Each step includes visualizations, explanations, and insights.
# 
# Data Source: [London Datastore – Greater London Authority](https://data.london.gov.uk/dataset/house-price-per-square-metre-in-england-and-wales)

# ## Step 1: Data Loading and Overview

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# Load the dataset
df = pd.read_csv("City_of_London_link_26122024.csv")


# In[3]:


# Overview of the dataset
df.info()
df.head()


# __Conclusion:__
# * 5836 rows and 16 columns
# * Key features: price, propertytype, duration, postcode, construction_age_band, dateoftransfer, and more.
# * Missing values were identified in the numberrooms and CONSTRUCTION_AGE_BAND columns.

# ## Step 2: Handling Missing Values

# In[4]:


# Filling missing values
df['numberrooms'].fillna(df['numberrooms'].mean(), inplace=True)
df['CONSTRUCTION_AGE_BAND'].fillna(df['CONSTRUCTION_AGE_BAND'].mode()[0], inplace=True)


# __Conclusion:__
# 
# Both columns were successfully filled using appropriate methods:
# 
# * numberrooms was filled using the mean value to maintain balance in continuous numerical data. Since the distribution of this column was not heavily skewed, using the mean ensures minimal impact on data integrity.
# 
# * CONSTRUCTION_AGE_BAND was filled with the mode (most frequent value), which is appropriate for categorical data and helps retain the most common construction period in the dataset.
# 
# These approaches ensured that no missing data remained in the dataset, improving data completeness for subsequent analyses.

# ## Step 3: Date Conversion and Feature Extraction
# 

# In[5]:


df['dateoftransfer'] = pd.to_datetime(df['dateoftransfer'])
df['transfer_year'] = df['dateoftransfer'].dt.year
df['transfer_month'] = df['dateoftransfer'].dt.month
df['transfer_day'] = df['dateoftransfer'].dt.day


# __Conclusion:__
# 
# The date conversion enables better time-based analysis, while the new features allow for more flexible trend identification.

# ## Step 4: Price Distribution Analysis
# 

# In[6]:


# Price distribution
df["price"].plot(kind = "hist", bins = 50, title = "Price Distribution")
plt.show()


# In[7]:


# Capping extreme values
upper_limit = 1357500
df['price'] = df['price'].clip(upper=upper_limit)


# __Conclusion:__
# 
# * Housing prices showed a right-skewed distribution.
# 
# * After capping, the dataset retained its core distribution while reducing the influence of extreme values.

# ## Step 5: Categorical Feature Analysis
# 

# In[8]:


# Property Type Distribution
df['propertytype'].value_counts().plot(kind='bar', title='Property Type Distribution')
plt.show()


# In[9]:


# Ownership Type Distribution
df['duration'].value_counts().plot(kind='bar', title='Ownership Type Distribution')
plt.show()


# In[10]:


# Extracting construction period
df['construction_period'] = df['CONSTRUCTION_AGE_BAND'].str.replace('England and Wales: ', '')


# In[11]:


# Visualizing construction periods
df['construction_period'].value_counts().plot(kind='bar', title='Construction Period Distribution')
plt.show()


# __Conclusion:__
# 
# * Flats were confirmed as the most common property type based on the frequency analysis.
# 
# * Leasehold ownership was confirmed to be significantly more common than Freehold.
# 
# * The most common construction periods were 1950-1966 and 1996-2002.

# ## Step 7: Impact of Ownership Type (duration) on Price

# In[12]:


# Ownership type price analysis
sns.boxplot(data=df, x='price', y='duration')
plt.show()


# __Conclusion:__
# 
# Freehold properties were significantly more expensive than Leasehold properties.

# ## Step 8: Time-Series Analysis

# In[13]:


# Time-series price trend analysis
price_trend = df.groupby('transfer_year')['price'].mean()
sns.lineplot(x=price_trend.index, y=price_trend.values, marker='o')
plt.show()


# __Conclusion:__
# 
# * Prices rose significantly from 2000 to 2008, followed by a sharp drop during the 2008 financial crisis.
# 
# * Prices peaked around 2016, likely influenced by Brexit and economic uncertainties.
# 
# * The post-2016 period showed volatility, with mixed price trends.

# ## Step 9: Correlation Analysis

# In[14]:


# Select only numerical columns
numeric_data = df.select_dtypes(include=['float64', 'int64'])

# Correlation matrix
correlation_matrix = numeric_data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()


# __Conclusion:__
# 
# * Strongest correlations with price were seen in:
# 
#     * priceper (+0.88)
# 
#     * tfarea (+0.57)
# 
# * numberrooms (+0.49)
# 
# Energy efficiency ratings showed minimal correlation with property prices.

# In[ ]:




