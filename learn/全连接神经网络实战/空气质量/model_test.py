import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import keras
from keras.models import load_model
from math import sqrt
from numpy import concatenate
from sklearn.metrics import mean_squared_error

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

dataset = pd.read_csv('data.csv')

sc = MinMaxScaler(feature_range=(0, 1))
scaled = sc.fit_transform(dataset)

dataset_sc = pd.DataFrame(scaled)

X = dataset_sc.iloc[:, :-1]
Y = dataset_sc.iloc[:, -1]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.02, random_state=42)

model = load_model('model.h5')

yhat = model.predict(x_test)

inv_yhat = concatenate((x_test, yhat), axis=1)
inv_yhat = sc.inverse_transform(inv_yhat)

prediction = inv_yhat[:, 6]

y_test = np.array(y_test)

y_test = np.reshape(y_test, (y_test.shape[0], 1))

inv_y = concatenate((x_test, y_test), axis=1)
inv_y = sc.inverse_transform(inv_y)
real = inv_y[:, 6]

rmse = sqrt(mean_squared_error(real, prediction))
mape = np.mean(np.abs((real - prediction) / prediction))

print('rmse', rmse)
print('mape', mape)

plt.plot(prediction, label='预测值')
plt.plot(real, label='真实值')
plt.title('全连接神经网络空气质量预测对比图')
plt.legend()
plt.show()
