import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import keras
from keras.layers import Dense
from keras.utils.np_utils import to_categorical
from sklearn.metrics import classification_report

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

dataset = pd.read_csv("breast_cancer_data.csv")

X = dataset.iloc[:, :-1]

Y = dataset['target']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

y_train_one = to_categorical(y_train, 2)
y_test_one = to_categorical(y_test, 2)

sc = MinMaxScaler(feature_range=(0, 1))
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

model = keras.Sequential()
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])

history = model.fit(x_train, y_train_one, epochs=110, batch_size=16,
                    verbose=2, validation_data=(x_test, y_test_one))
model.save('model.h5')

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.title("全连接神经网络loss值图")
plt.legend
plt.show()


plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.title("全连接神经网络accuracy值图")
plt.legend
plt.show()
