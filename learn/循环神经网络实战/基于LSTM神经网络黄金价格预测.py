import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, LSTM
import keras

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 加载历史数据文件
dataset = pd.read_csv('LBMA-GOLD.csv', index_col='Date')

# 设置训练集长度
training_len = 1256 - 200

# 获取训练集数据
training_set = dataset.iloc[0:training_len, [0]]

# 获取测试集数据
test_set = dataset.iloc[training_len:, [0]]

# 将数据归一化，方便神经网络训练
sc = MinMaxScaler(feature_range=(0, 1))
train_set_scaled = sc.fit_transform(training_set)
test_set = sc.transform(test_set)

x_train = []
y_train = []

x_test = []
y_test = []

# 利用for循环，遍历整个训练集，提取训练集中连续5个采样点的数据作为输入特征x_train，第6个采样点的数据作为标签
for i in range(5, len(train_set_scaled)):
    x_train.append(train_set_scaled[i - 5:i, 0])
    y_train.append(train_set_scaled[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0], 5, 1))

for i in range(5, len(test_set)):
    x_test.append(test_set[i - 5:i, 0])
    y_test.append(test_set[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

x_test = np.reshape(x_test, (x_test.shape[0], 5, 1))

# 搭建神经网络模型
model = keras.Sequential()
model.add(LSTM(80, return_sequences=True, activation='relu'))
model.add(LSTM(100, return_sequences=False, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

# 对模型进行编译，选用学习率为0.01
model.compile(loss='mse', optimizer=keras.optimizers.Adam(0.01))

# 将训练集和测试集放入网络进行训练，每批次送入的数据为32个数据，一共训练50轮，将测试集样本放入到神经网络中测试其验证集的loss值
history = model.fit(x_train, y_train, batch_size=32, epochs=100, validation_data=(x_test, y_test))
model.save('model.h5')

# 绘制训练集和测试集的loss值对比图
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.title('LSTM神经网络loss值')
plt.legend()
plt.show()

