import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import keras

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 导入数据集
dataset = pd.read_csv("data.csv")
# print(dataset)

# 将数据进行归一化
sc = MinMaxScaler(feature_range=(0, 1))
scaled = sc.fit_transform(dataset)
# print(scaled)

# 将归一化好的数据转化为datafrme格式，方便后续处理
dataset_sc = pd.DataFrame(scaled)
# print(dataset_sc)

# 将数据集中的特征和标签找出来
X = dataset_sc.iloc[:, :-1]
Y = dataset_sc.iloc[:, -1]

# 划分数据集
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 利用keras搭建神经网络模型
model = keras.Sequential()
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

# 对神经网络模型进行编译
model.compile(loss='mse', optimizer='SGD')

# 进行模型的训练
history = model.fit(x_train, y_train, epochs=100, batch_size=24, verbose=2, validation_data=(x_test, y_test))
model.save("model.h5")

# 绘制模型的训练和验证集的loss值对比图
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.title("全连接神经网络loss值图")
plt.legend()
plt.show()
