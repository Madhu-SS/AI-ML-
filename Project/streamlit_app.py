import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import acf,pacf,adfuller
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from scipy import stats
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import keras
import tensorflow
from keras.layers import Dense,Dropout
from keras.models import Sequential
from keras.datasets import mnist
from keras.optimizers import RMSprop,Adam
from tensorflow.keras.layers import LSTM,Bidirectional

data=pd.read_excel("Company stock prices.xlsx")
df2=data[['Date','Close']]
df=data[['Date','Close']]
df.set_index('Date',inplace=True)
from sklearn.preprocessing import MinMaxScaler
min_max=MinMaxScaler()
df1=min_max.fit_transform(np.array(df))
training_size=int(len(df1)*0.75)
test_size=len(df1)-training_size
train_data,test_data=df1[0:training_size,:],df1[training_size:,:]
def create_dataset(dataset,time_step=1):
    dataX,dataY=[],[]
    for i in range(len(dataset)-time_step):
        a=dataset[i:(i+time_step),0]
        dataX.append(a)
        dataY.append(dataset[i+ time_step,0])
    return np.array(dataX),np.array(dataY)
time_step=50
x_train,y_train=create_dataset(train_data,time_step)
x_test,y_test=create_dataset(test_data,time_step)
x_train=x_train.reshape(514,50,1)
x_test=x_test.reshape(139,50,1)

# Load the saved LSTM model
model = keras.models.load_model("my_work.h5")
pred=model.predict(x_test)
predd=min_max.inverse_transform(pred)
y_testt=y_test.reshape(-1,1)
y_testt=min_max.inverse_transform(y_testt)
y_testt=pd.DataFrame(y_testt)
predd=pd.DataFrame(predd)
d=data[['Date']][615:-1]
dd=d.reset_index()

predicted_price = []

def stock_prediction(days):
    t = x_test[-1]
    tt = t.reshape(1, 50, 1)
    p = model.predict(tt)
    predicted_price.append(p[0])

    for i in range(days-1):    
        stockValue_list = list(t)
        stockValue_list.append(p[0])
        testing_array = np.array(stockValue_list[-50:])
        p = model.predict(testing_array.reshape(1, 50, 1))
        predicted_price.append(p[0])
    predicted_pricee=min_max.inverse_transform(predicted_price)
    return predicted_pricee
b=stock_prediction(263)
# Assuming you already have a DataFrame 'stock_prices' with 'Date' and 'Close' columns
# If not, you can create it from your existing data

# Convert the 'Date' column to a datetime object if it's not already
df2['Date'] = pd.to_datetime(df2['Date'])

# Find the latest date in your DataFrame
latest_date = df2['Date'].max()

# Calculate the date one year from the latest date
one_year_later = latest_date + pd.DateOffset(years=1)

# Create a date range from the latest date to one year later, including only weekdays
extended_date_range = pd.date_range(start=latest_date, end=one_year_later, freq='B')

# Create a DataFrame with the extended date range
extended_stock_prices = pd.DataFrame({'Date': extended_date_range})

# Merge the extended date DataFrame with your original stock price data
extended_stock_prices = pd.merge(extended_stock_prices, df2, on='Date', how='left')
date_col=extended_stock_prices[["Date"]]
new_columns={0:"Close"}
predicted_df=pd.DataFrame(b)
predicted_df=predicted_df.rename(columns=new_columns)
dfff=pd.concat([date_col,predicted_df],axis=1)
dfff=dfff.rename(columns=new_columns)

# Streamlit UI
st.title("LSTM Model Deployment")
date=st.date_input("ENTER YOUR DATE HERE")
def _stock_p(date):
    resu=dfff[(dfff['Date'])==(date)]
    return resu
sto=_stock_p(date)
st.write(sto)
