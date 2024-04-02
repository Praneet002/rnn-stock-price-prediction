# Stock Price Prediction

## AIM

To develop a Recurrent Neural Network model for stock price prediction.

## Problem Statement and Dataset


## Design Steps

### Step 1:
Read the csv file and create the Data frame using pandas.

### Step 2:
Select the " Open " column for prediction. Or select any column of your interest and scale the values using MinMaxScaler.

### Step 3:
Create two lists for X_train and y_train. And append the collection of 60 readings in X_train, for which the 61st reading will be the first output in y_train.

### Step 4:
Create a model with the desired number of neurons and one output neuron.

### Step 5:
Follow the same steps to create the Test data. But make sure you combine the training data with the test data.

### Step 6:
Make Predictions and plot the graph with the Actual and Predicted values.

## Program
#### Name:Praneet S
#### Register Number:212221230078
```
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras import layers
from tensorflow.keras import models
from keras.models import Sequential
dataset_train = pd.read_csv('trainset (1).csv')
dataset_train.columns
dataset_train.head()
train_set = dataset_train.iloc[:,1:2].values
type(train_set)
train_set.shape
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(train_set)
training_set_scaled.shape
X_train_array = []
y_train_array = []
for i in range(60, 1259):
  X_train_array.append(training_set_scaled[i-60:i,0])
  y_train_array.append(training_set_scaled[i,0])
X_train, y_train = np.array(X_train_array), np.array(y_train_array)
X_train1 = X_train.reshape((X_train.shape[0], X_train.shape[1],1))
X_train.shape
length = 60
n_features = 1
model = models.Sequential()
model.add(layers.SimpleRNN(10,input_shape=(length,n_features)))
model.add(layers.Dense(1))
model.compile(optimizer='rmsprop', loss='mse',metrics ='accuracy')
print("Name:Praneet S      Register Number:212221230078          ")
model.summary()
model.fit(X_train1,y_train,epochs=100, batch_size=32)
dataset_test = pd.read_csv('testset (1).csv')
test_set = dataset_test.iloc[:,1:2].values
test_set.shape
dataset_total = pd.concat((dataset_train['Open'],dataset_test['Open']),axis=0)
inputs = dataset_total.values
inputs = inputs.reshape(-1,1)
inputs_scaled=sc.transform(inputs)
X_test = []
for i in range(60,1384):
  X_test.append(inputs_scaled[i-60:i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test,(X_test.shape[0], X_test.shape[1],1))
X_test.shape
predicted_stock_price_scaled = model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price_scaled)
print("Name:Praneet S           Register Number:212221230078     ")
plt.plot(np.arange(0,1384),inputs, color='red', label = 'Test(Real) Google stock price')
plt.plot(np.arange(60,1384),predicted_stock_price, color='blue', label = 'Predicted Google stock price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
```
## Output

### True Stock Price, Predicted Stock Price vs time

![image](https://github.com/Praneet002/rnn-stock-price-prediction/assets/94308200/bf22a1a3-0772-47e1-9f8e-ee540f4d3bd3)

### Mean Square Error

![image](https://github.com/Praneet002/rnn-stock-price-prediction/assets/94308200/e34c654d-d3f1-4600-8922-71ec1d41cb0f)

## Result

Thus, we have successfully created a Simple RNN model for Stock Price Prediction.

