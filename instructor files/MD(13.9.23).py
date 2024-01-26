#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import streamlit as st


# In[3]:


st.title('Model Deployment: Logistic Regression')
st.sidebar.header('User Input Parameters')


# In[21]:


def user_input_parameters():
    CLMSEX=st.sidebar.selectbox('Gender',('1','0'))
    CLMINSUR=st.sidebar.selectbox('Insurance',('1','0'))
    SEATBELT=st.sidebar.selectbox('Seatbelt',('1','0'))
    CLMAGE=st.sidebar.number_input('Insert the Age')
    LOSS= st.sidebar.number_input('Inser_loss')
    data= {'CLMSEX':CLMSEX,
           'CLMINSUR':CLMINSUR,
           'SEATBELT':SEATBELT,
           'CLMAGE':CLMAGE,
           'LOSS':LOSS}
    features=pd.DataFrame(data,index=[0])
    return features
df= user_input_parameters()
st.subheader('User Input Parameters')
st.write(df)
claimants= pd.read_csv('claimants.csv')
claimants.drop('CASENUM',axis=1,inplace=True)
claimants.dropna(inplace=True)
x= claimants.iloc[:,1:6]
y= claimants.iloc[:,0]
clf= LogisticRegression()
clf.fit(x,y)
prediction= clf.predict(df)
prediction_proba= clf.predict_proba(df)

st.subheader('Predicted Result')
st.write('Yes' if prediction_proba[0][1]>0.5 else 'No')

st.subheader('Predicted Probability')
st.write(prediction_proba)


# In[ ]:




