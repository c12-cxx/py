import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import keras
from keras.layers import Dense
from keras.utils.np_utils import to_categorical
from sklearn.metrics import classification_report
from keras.models import load_model

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

dataset = pd.read_csv("breast_cancer_data.csv")

X = dataset.iloc[:, :-1]

Y = dataset['target']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

y_test_one = to_categorical(y_test, 2)

sc = MinMaxScaler(feature_range=(0, 1))
x_test = sc.fit_transform(x_test)

model = load_model('model.h5')

predict = model.predict(x_test)

y_pred = np.argmax(predict, axis=1)

result = []
for i in range(len(y_pred)):
    if y_pred[i] == 0:
        result.append('良性')
    else:
        result.append('恶性')

report = classification_report(y_test, y_pred, labels=[0, 1], target_names=['良性', '恶性'])
print(report)
