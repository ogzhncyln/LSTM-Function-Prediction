# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Sequential
import numpy as np
import matplotlib.pyplot as plt
import math

def scaler(arr):
  min = arr.min()
  max = arr.max()
  return min,max,(arr - min) / (max - min)

def descaler(arr,min,max):
  return arr * (max - min) + min

def Func(x):
  return math.sin(x/5) + 5*math.cos(x/5) ** 2

input_size = 50
output_size = 50
source_length = 110.0
step_size = 0.1

x_source = np.arange(0,source_length,step_size)
min,max,y_source = scaler(np.array([Func(x) for x in x_source]))

plt.plot(x_source,descaler(y_source,min,max))
plt.show()

def CreateLSTMData(x_source_arr,y_source_arr,input_size,output_size):
  x = []
  y = []
  for i in range(len(x_source_arr)- input_size - output_size):
    x.append(x_source_arr[i:i+input_size])
    y.append(y_source_arr[i+input_size:i+input_size+output_size])
  x = np.reshape(x,(len(x),input_size,1))
  y = np.reshape(y,(len(y),output_size))
  return x,y

x_train , y_train = CreateLSTMData(x_source,y_source,input_size,output_size)

model = Sequential()
model.add(LSTM(100, input_shape=(x_train.shape[1], x_train.shape[2])))
model.add(Dense(y_train.shape[1]))
model.compile(loss='mse', optimizer='adam')

model.fit(x_train,y_train,epochs=500,verbose=2)

model.save("model.h5")

x_t = np.arange(0,source_length,step_size)
min_t,max_t,y_t = scaler(np.array([Func(x) for x in x_t]))
x_test , y_test = CreateLSTMData(x_t,y_t,input_size,output_size)
predictions = descaler(model.predict(x_test),min_t,max_t)

for i in range(len(y_t)):
  plt.plot(descaler(y_test[i],min_t,max_t),label = "real value")
  plt.plot(predictions[i],label = "predicted values")
  plt.legend()
  plt.show()