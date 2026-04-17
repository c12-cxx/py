import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

dataset = pd.read_csv("breast_cancer_data.csv")

X = dataset.iloc[:, :-1]

Y = dataset['target']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

sc = MinMaxScaler(feature_range=(0, 1))
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

lr = LogisticRegression()
lr.fit(x_train, y_train)

# print('w:', lr.coef_)
# print('b:', lr.intercept_)

pre_result = lr.predict(x_test)

pre_result_proba = lr.predict_proba(x_test)
# print(pre_result_proba)

pre_list = pre_result_proba[:, 1]

threshold = 0.3

result = []
result_name = []

for i in range(len(pre_list)):
    if pre_list[i] > threshold:
        result.append(1)
        result_name.append('恶性')
    else:
        result.append(0)
        result_name.append('良性')

# print(result)
#
# print(result_name)

report = classification_report(y_test,result,labels=[0,1],target_names=['良性','恶性'])
print(report)