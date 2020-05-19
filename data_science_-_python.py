#!/usr/bin/env python
# coding: utf-8

# <h2>
# <br>Project Title: “Melbourne house price data"
# <br>Name: “Vikash Pateshwari"
# <br>Email: “vikash.pateshwari@wipro.com"
# <br>Company : “Wipro Technology”
# <br>output: html_document , .ipynb , .py
# </h2>

# <h2>Business Understanding </h2>
#  <h3>Melbourne house price data</h3>
# 

# <h2>Data Understanding</h2>
# <h4>You can download the data from https://www.kaggle.com/anthonypino/melbourne-housing-market#Melbourne_housing_FULL.csv
# <br>You can understand the data by looking at the data dictionary provided @ https://rpubs.com/kunaljubce/mlb_housing_data
# </h4>

# <h4><ol>
# <li>Import the data set in Python.</li>
# <li>View the dataset</li>
# <li>See the structure and the summary of the dataset to understand the data.</li>
# <li>Find out the number of:</li>
# 	 <ul>
#      <li>Numeric attributes:</li>
# 	 <li>Categorical attributes:</li>
#      </ul>
#     </h4>

# In[1]:


#1.Import the data set in Python.
#downloaded the data sets
import numpy as np
import pandas as pd
full = pd.read_csv(r'C:\Users\imvik\Wipro Assignment\AI\data_science_-_python\melbourne-housing-market\Melbourne_housing_FULL.csv')


# In[2]:


#2.View the dataset 
#head for top 5 rows or tail for 5 last items
#full.tail()
full.head()


# In[3]:


#See the structure and the summary of the dataset to understand the data.
full.info()


# In[4]:


full.shape


# In[5]:


full.columns


# In[6]:


full.describe()


# In[7]:


#Find out the number of:
#    Numeric attributes:
numeric = full.select_dtypes(exclude='object')
len(numeric.columns)


# In[8]:


#    Categorical attributes:
categorical= full.select_dtypes(include='object')
len(categorical.columns)


# <h2> Data Preparation : Data Cleaning</h2>
# <ul>
# <li>Duplicate values: Identify if the datasets have duplicate values or not and remove the duplicate values. </li>
#     <li>Find out the number of rows present in the dataset</li>
# <li>Before removing duplicate values</li>
# <li>After removing duplicate values</li>
# <li>Variable type: Check if all the variables have the correct variable type, based on the data dictionary. If not, then change them.</li>
# 	<li>For how many attributes did you need to change the data type?</li>
# <li>Missing value treatment: Check which variables have missing values and use appropriate treatments. </li>
# 	<li>	For each of the variables, find the number of missing values and provide the value that they have been imputed with.</li>
# <li>Outlier Treatment: </li>
# 	<li>Identify the varibales : Make a subset of the dataset with all the numeric variables. </li>
# 	<li>Outliers : For each variable of this subset, carry out the outlier detection. Find out the percentile distribution of each variable and carry out capping and flooring for outlier values. </li> 
# </ul>

# In[9]:


#Duplicate values: Identify if the datasets have duplicate values or not and remove the duplicate values.
full[full.duplicated()]


# In[10]:


#Find out the number of rows present in the dataset
#Before removing duplicate values
duplicate = full.duplicated()
duplicate.count()


# In[11]:


#After removing duplicate values
remdupl = full.drop_duplicates()
len(remdupl)


# In[12]:


#Variable type: Check if all the variables have the correct variable type,
#               based on the data dictionary. If not, then change them.
full.dtypes


# In[13]:


#as object is not  datatype so it need to converted in cateogical data type.
objectdtye = full.select_dtypes({object}).columns
objectdtye


# In[14]:


full[objectdtye] = full[objectdtye].astype('category')


# In[15]:


#For how many attributes did you need to change the data type?
full.info()
#from below info we can check there are 8 categorical data.
#dtypes: category(8), float64(12), int64(1)


# In[16]:


#Missing value treatment: Check which variables have missing values and use appropriate treatments.
#For each of the variables, find the number of missing values and provide the value that they have been imputed with.
full.isnull().sum()


# In[17]:


#droping missing value columns 
full1 = full
len(full1), len(full1.dropna())


# In[18]:


#method 2 imputing the values
full1.isna().sum()


# In[19]:


Price_mean = full1.Price.mean()
Distance_mean = full1.Distance.mean()
Postcode_mean = full1.Postcode.mean()
Bedroom2_mean = full1.Bedroom2.mean()
Bathroom_mean = full1.Bathroom.mean()
Car_mean = full1.Car.mean()
Landsize_mean = full1.Landsize.mean()
BuildingArea_mean = full1.BuildingArea.mean()
YearBuilt_mean = full1.YearBuilt.mean()
Lattitude_mean = full1.Lattitude.mean()
Longtitude_mean = full1.Longtitude.mean()
Propertycount_mean = full1.Propertycount.mean()
full1.fillna(value = {'Price':Price_mean
                      ,'Distance':Distance_mean
                      ,'Postcode':int(Postcode_mean)
                      ,'Bedroom2':Bedroom2_mean
                      ,'Car':Car_mean
                      ,'Bathroom':Bathroom_mean
                      ,'Landsize':Landsize_mean
                      ,'BuildingArea':BuildingArea_mean
                      ,'YearBuilt':int(YearBuilt_mean)
                      ,'Lattitude':Lattitude_mean
                      ,'Longtitude':Longtitude_mean
                      ,'Propertycount':Propertycount_mean
                      ,'CouncilArea' : 'Boroondara City Council'#we are taking most used categroical value_counts()
                      ,'Regionname' :'Southern Metropolitan'} #we are taking most used categroical value_counts() 
                      ,inplace = True
                      )
#full1["Price"].fillna( method ='ffill')


# In[20]:


full1.isna().sum()


# In[21]:


#missing no library offers a very nice way to visualize the distribution of NaN values.
import missingno as msno
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
msno.bar(full1, figsize=(12, 6), fontsize=12, color='steelblue')


# In[24]:


#Outlier Treatment:
#Identify the varibales : Make a subset of the dataset with all the numeric variables.
full1_numeric=full1.select_dtypes(['int64','float64'])
full1_numeric.head()


# In[27]:


#Outliers : For each variable of this subset, carry out the outlier detection.
#          Find out the percentile distribution of each #variable and carry out capping and flooring for outlier values
full1_numeric.describe()
#75% is Q3 
#25% is Q1 
#IQR=Q3-Q1 
#floor = Q1 -1.5*IQR 
#cap=Q3+1.5*IQR
#we can also use full1_numeric['Rooms'].quantile(0.25) to get Q1 Value.


# In[29]:


#BoxPlot for variable "Rooms"
import seaborn as sns
sns.boxplot(x=full1_numeric['Rooms'])


# In[32]:


sns.boxplot(x=full1_numeric['Price'])


# In[33]:


sns.boxplot(x=full1_numeric['Distance'])


# In[35]:


sns.boxplot(x=full1_numeric['Bedroom2'])


# In[36]:


sns.boxplot(x=full1_numeric['Bathroom'])


# In[37]:


sns.boxplot(x=full1_numeric['Car'])


# In[38]:


sns.boxplot(x=full1_numeric['Landsize'])


# In[39]:


sns.boxplot(x=full1_numeric['BuildingArea'])


# In[40]:


sns.boxplot(x=full1_numeric['Propertycount'])


# In[41]:


for col in full1_numeric:
    if col!='Lattitude' and col!='Longtitude':# not considering Lattitude & Longitude as Outlier treatment
        print("\n\nOutlier for ",col)
        # Cap=IQ3+1.5*IQR & Floor=1.5*IQR-IQ1
        Q1 = full1_numeric[col].quantile(0.25)
        Q3 = full1_numeric[col].quantile(0.75)
        IQR = Q3 - Q1
        print("IQR for ",col,"=",IQR)
        limit=1.5*IQR
        floor = limit - Q1
        print("Floor for ",col,"=",floor)
        Cap = Q3 + limit
        print("Cap for ",col,"=",Cap)
        #values less than (1.5*IQR-Q1) and more than (1.5*IQR+Q3) are removed.
        #Removing Outlier values for Price due to Large number
        if col=='Price':
            full1_numeric=full1_numeric[(full1_numeric[col]>=floor)&(full1_numeric[col]<=Cap)]
            print(full1_numeric.shape)


# <h2>Data Preparation Feature Engineering:</h2>
# <h3>Feature Transformation:</h3>
# <h4>Identify variables that have non-linear trends.
# <br>How many variables have non-linear trends?
# <br>Transform them (as required)
# </h4>

# In[42]:


#Identify variables that have non-linear trends
sns.jointplot(x='Rooms',y='Price',data=full1,kind='scatter')


# In[43]:


sns.jointplot(x='Bedroom2',y='Price',data=full1,kind='scatter')


# In[44]:


sns.jointplot(x='Bathroom',y='Price',data=full1,kind='scatter')


# In[45]:


sns.jointplot(x='Car',y='Price',data=full1,kind='scatter')


# In[46]:


sns.jointplot(x='Landsize',y='Price',data=full1,kind='scatter')


# In[47]:


sns.jointplot(x='BuildingArea',y='Price',data=full1,kind='scatter')


# In[49]:


# Matrix form for correlation data
full1.corr()


# In[53]:


sns.heatmap(full1.corr())
#Rooms, Bedroom2, Bathroom & Car have weak positive correlation with Price.
#Distance have weak weak negative correlation with Price
#Landsize & Building Area do not have much correlation with Price and it is having non-linear trend.


# <h2>Standardization:</h2>
# <h3>Name the variables to be standardised before using a distance-based algorithm</h3>

# In[56]:


#All numeric variables is standardised before applying KNN method(distance based algorithm)
#Z-Score
int_col=['Rooms','Price','Distance','Bedroom2','Bathroom','Car','Landsize','BuildingArea','YearBuilt','Lattitude','Longtitude']
from scipy.stats import zscore
full1_z=full1[int_col].apply(zscore)
full1_z


# <h2>Dummy encoding :</h2>
# <h4>Identify the number of dummy variables to be created for the variable steel.
# <br>Submit the Python script with Outputs in a document
# </h4>

# In[59]:


full1.Suburb.value_counts()

