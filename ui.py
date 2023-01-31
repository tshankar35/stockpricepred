import math
from pandas_datareader import data as pdr
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM  
import yfinance as yfin
import plotly.graph_objs as go
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import tensorflow as tf
from google.colab import drive
drive.mount('/content/drive')
def fetchdata(ticker):
  yfin.pdr_override()
  df = pdr.get_data_yahoo(ticker,start='2012-01-01')
  return df

def stockvisualiser(df,name):
  fig = go.Figure()
  fig.add_trace(go.Candlestick(x=df.index,open=df['Open'],high=df['High'],low=df['Low'],close=df['Close'],name='market data'))
  fig.update_layout(title=name,yaxis_title='Stock Price (INR Per Share')
  fig.show()



def preprocess(df):
  #Create New DataFrame Only with Close Column
  data = df.filter(['Close'])
  #Convert the dataframe to Numpy array
  dataset = data.values
  #Get the number of rows to train the model on
  training_data_len = math.ceil(len(dataset)*0.8)

  #Scale the Data
  scaler = MinMaxScaler (feature_range=(0,1))
  scaled_data = scaler.fit_transform(dataset)

  #Create the training data set
  #Create the scaled training dataset

  train_data = scaled_data[0:training_data_len, :]
  #Split the data into x_train and y_train datsets

  x_train = []
  y_train = []

  for i in range(60,len(train_data)):
    x_train.append(train_data[i-60:i,0])
    y_train.append(train_data[i,0])

  #Convert the x_train and y_train to numpy arrays
  x_train, y_train = np.array(x_train), np.array(y_train)

  #Reshape the data
  x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))

  #Creating testing data set
  #Create a new array containing scaled values 
  test_data = scaled_data[training_data_len-60:,:]
  #Create datasets x_test and y_test
  x_test = []
  y_test = dataset[training_data_len:,:]
  for i in range(60,len(test_data)):
    x_test.append(test_data[i-60:i,0])
  
  #Convert the data to a numpy array
  x_test = np.array(x_test)

  #Reshape the data
  x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))

  return x_train,y_train,x_test,y_test,scaler

def modelinitaliser(x_train):
  model = Sequential()
  model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1],1)))
  model.add(LSTM(50, return_sequences=False))
  model.add(Dense(25))
  model.add(Dense(1))
  return model

def modeltrainer(x_train,y_train,model):
  #Compile the model
  model.compile(optimizer='adam',loss='mean_squared_error')
  #Train
  model.fit(x_train,y_train, batch_size=1, epochs=1)
  return model

def modeltester(x_test,y_test,model,scaler):
  predictions = model.predict(x_test)
  predictions = scaler.inverse_transform(predictions)
  rmse = np.sqrt(np.mean(predictions-y_test)**2)
  return rmse

def plotpredictions(data,training_data_len,predictions):
  #Plot the data
  train = data[:training_data_len]
  validation = data[training_data_len: ]
  validation['Predictions'] = predictions
  #Visualise
  plt.figure(figsize=(16,8))
  plt.title('Model')
  plt.xlabel('Data',fontsize=18)
  plt.ylabel('Close Price Rupees',fontsize=18)
  plt.plot(train['Close'])
  plt.plot(validation[['Close','Predictions']])
  plt.legend(['Train','Validation','Predictions'],loc='lower right')
  plt.show()

def nextdatprediction(ticker,scaler,model):
  #Get the quote
  quote = pdr.get_data_yahoo(ticker,start='2012-01-01')
  new_df = quote.filter(['Close'])
  #Get the last 60 day closing values and convert the dataframe to an array
  last_60_days = new_df[-60:].values
  #Scale the data to be values between
  last_60_days_scaled = scaler.transform(last_60_days)
  #Create an empty list
  X_test = []
  #Append the past 60 days
  X_test.append(last_60_days_scaled)
  #Convert the X_test dataset to numpy array
  X_test = np.array(X_test)
  #Reshape the Data
  X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
  #Get the Predicted Scaled Price
  pred_price = model.predict(X_test)
  #Undo the Scaling
  pred_price = scaler.inverse_transform(pred_price)
  return pred_price

def exporter(name,model):
  filename = name+'.sav'
  localhost_save_option = tf.saved_model.SaveOptions(experimental_io_device="/job:localhost")
  model.save(filename, options=localhost_save_option)

def main():
  df_stock = pd.read_csv('/content/drive/MyDrive/nse.csv')
  for i in range(1):
    try:
      pricelist = fetchdata(df_stock['BSE SYMBOL'][i])
      x_train,y_train,x_test,y_test,scaler=preprocess(pricelist)
      print("Training: ",df_stock['Company Name'][i])
      print("Epoch 1")
      model = modelinitaliser(x_train)
      model = modeltrainer(x_train,y_train,model)
      rmse = modeltester(x_test,y_test,model,scaler)
      print("Current Epoch RMSE: ",rmse)
      k=2
      while rmse>10:
        print("Epoch ",k)
        model = modeltrainer(x_train,y_train,model)
        rmse = modeltester(x_test,y_test,model,scaler)
        if k>20 and rmse<30:
          print("Current Epoch RMSE: ",rmse)
          break
        k+=1
        print("Current Epoch RMSE: ",rmse)
      print("Final RMSE is: ",rmse)
      symbol = str(df_stock['BSE SYMBOL'][i])
      symbol = symbol[:-3]
      exporter('/content/drive/MyDrive/models/'+symbol,model)
    except:
      continue

main()



