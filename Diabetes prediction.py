#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing some libraries


# In[2]:


import pandas as pd

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split 
from sklearn import svm 
from sklearn.metrics import accuracy_score


# In[3]:


#read the datasets from csv
#PIMA Diabetees datasetset


# In[4]:


data = pd.read_csv("C:\\Users\\hp\\Downloads\\diabeteesdatabase\\diabetes.csv")


# In[5]:


data


# In[6]:


data.shape


# In[7]:


data.info()


# In[8]:


data.columns


# In[9]:


#check how many values are null?


# In[10]:


data.isnull().sum()


# In[11]:


#check the outcomes how many person are diabetic or non-diabetic
#0-Nondiabetic
#1-Diabetic


# In[12]:


data["Outcome"].value_counts()


# In[13]:


data.groupby("Outcome").mean()


# In[14]:


X=data.drop(columns="Outcome",axis=1)
Y=data["Outcome"]


# In[15]:


#separting the data and labels


# In[16]:


X


# In[17]:


Y


# In[18]:


#standarize the data


# In[19]:


s=StandardScaler()


# In[20]:


s.fit(X)


# In[21]:


standardized=s.transform(X)


# In[22]:


standardized


# In[23]:


#train and test the data with the test size 0.2


# In[24]:


X_test,X_train,Y_test,Y_train=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)


# In[25]:


#Training the model


# In[26]:


classify=svm.SVC(kernel='linear')


# In[27]:


classify.fit(X_train,Y_train)


# In[28]:


#Acuuracy score


# In[29]:


prediction=classify.predict(X_train)
accuracy=accuracy_score(prediction,Y_train)


# In[30]:


print("The accuracy of the model(train data) is ","{:.3f}".format(accuracy*100),"%")


# In[31]:


prediction1=classify.predict(X_test)
accuracy1=accuracy_score(prediction1,Y_test)


# In[32]:


print("The accuracy of the model(test data) is ","{:.3f}".format(accuracy1*100),"%")


# In[33]:


input_data = (5,166,72,20,116,25.8,0.257,51)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# standardize the input data
std_data = s.transform(input_data_reshaped)


prediction = classify.predict(std_data)


if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')


# In[ ]:




