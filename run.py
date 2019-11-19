
import pandas as pd
import numpy as np
from keras.models import Model, Sequential
import os, sys
import glob
import matplotlib.pyplot as plt
import csv
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import BatchNormalization
from keras.objectives import MSE, MAE
from keras.callbacks import EarlyStopping
from sklearn import metrics


def main():
  la_df = pd.read_csv('./data/lacity/500_cities_lacity_mental_health.csv')
  X = pd.read_csv('la_vgg.csv').sort_values(by=['TractFIPS']).reset_index().drop(["index"], axis=1)
  target = la_df.sort_values(by="TractFIPS").reset_index().drop(["index"], axis=1)
  target = target[target.TractFIPS != 6037930401]
  y = target.Data_Value
  X = X.drop(['TractFIPS'], axis=1)

  test_mem_df = pd.read_csv('mem_vgg.csv')
  test_mem_df = test_mem_df.sort_values(by=['TractFIPS']).reset_index().drop(['index'], axis=1)
  mem_df = pd.read_csv('./data/Memphis/500_cities_memphis_mental_health.csv').sort_values(by=['TractFIPS']).reset_index().drop(['index'], axis=1)
  test_mem_df = test_mem_df.drop(['TractFIPS'], axis=1)
  X_mem = test_mem_df
  y_mem = mem_df['Data_Value']
  
  model = Sequential()
  model.add(Dense(input_dim=X.shape[1], units=32))
  model.add(BatchNormalization())
  model.add(Activation("relu"))
  model.add(Dense(512))
  model.add(Activation("relu"))
  model.add(Dropout(0.50))
  model.add(Dense(256))
  model.add(Activation("relu"))
  model.add(Dense(units=1))
  model.compile(loss='mean_squared_error', optimizer='adam')

  early_stopping = EarlyStopping(monitor='val_loss', patience=30)
  train_log = model.fit(X, y, batch_size=16, epochs=200, validation_split=0.3, verbose=2, callbacks=[early_stopping])
  pred_y = model.predict(X_mem)

  print('Mean Absolute Error:', metrics.mean_absolute_error(y_mem, pred_y))  
  print('Mean Squared Error:', metrics.mean_squared_error(y_mem, pred_y))  
  print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_mem, pred_y)))
  print('R2 score:', metrics.r2_score(y_mem, pred_y))  

  result = pd.DataFrame(pred_y)
  result.to_csv("submission.csv")

if __name__ == "__main__":
  main()