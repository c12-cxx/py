# import numpy as np
#
# x_data = [1.0, 2.0, 3.0]
# y_data = [2.0, 4.0, 6.0]
#
# w = 0.1  # 适当小的初始化值
#
#
# def forward(x):
#     return x * w
#
#
# def cost(xs, ys):
#     cost = 0
#     for x, y in zip(xs, ys):
#         y_pred = forward(x)
#         cost += (y_pred - y) ** 2
#     return cost / len(xs)
#
#
# def gradient(xs, ys):
#     grad = 0
#     for x, y in zip(xs, ys):
#         grad += 2 * x * (x * w - y)
#     return grad / len(xs)
#
#
# print('Predict(before training)', 4, forward(4))
#
# # 训练过程
# for epoch in range(500):
#     cost_val = cost(x_data, y_data)
#     grad_val = gradient(x_data, y_data)
#     w -= 0.01 * grad_val  # 可能需要调整学习率
#     if epoch % 50 == 0:  # 每50次输出一次结果
#         print(f'Epoch: {epoch}, w={w:.4f}, loss={cost_val:.4f}')
#
# print('Predict(after training)', 4, forward(4))

#
# import pandas as pd
# import numpy as np
#
# # 设置随机种子以保证结果可复现
# np.random.seed(2025)
# n_samples = 500
#
# # 模拟核心特征
# data = pd.DataFrame({
# ‘面积_平方米‘: np.random.normal(90, 20, n_samples).clip(50, 200),  # 均值90，标准差20
# ‘卧室数量‘: np.random.choice([1, 2, 3, 4], n_samples, p=[0.2, 0.5, 0.2, 0.1]),
# ‘距市中心距离_公里‘: np.random.exponential(5, n_samples).clip(1, 20),
# ‘楼龄_年‘: np.random.randint(1, 30, n_samples),
# ‘是否有地铁‘: np.random.binomial(1, 0.6, n_samples)  # 60%的房源靠近地铁
# })
#
# # 模拟房价（单位：万元）：基于特征构造一个非线性关系，并加入随机噪声
# # 假设房价主要由面积、距离和楼龄决定，并存在交互效应
# base_price = (data[‘面积_平方米‘] * 0.8 +
#                    100 / (data[‘距市中心距离_公里‘] + 1) -
#              data[‘楼龄_年‘] *0.5 +
#                               data[‘是否有地铁‘] *15 +
#                                                   data[‘卧室数量‘] *5)
# noise = np.random.normal(0, 15, n_samples)  # 加入随机噪声
# data[‘房价_万元‘] = (base_price + noise).clip(80, 600)  # 确保价格在合理范围
#
# print(data.head())
# print(f"\n数据集形状: {data.shape}")
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error, r2_score
#
# # 1. 准备特征(X)和目标变量(y)
# X = data.drop(‘房价_万元‘, axis = 1)
# y = data[‘房价_万元‘]
#
# # 2. 划分训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
#
# # 3. 标准化数值特征（树模型对标准化不敏感，但为了公平比较，统一处理）
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)
#
# # 4. 训练与评估线性回归模型
# lr_model = LinearRegression()
# lr_model.fit(X_train_scaled, y_train)
# y_pred_lr = lr_model.predict(X_test_scaled)
# mse_lr = mean_squared_error(y_test, y_pred_lr)
# r2_lr = r2_score(y_test, y_pred_lr)
#
# # 5. 训练与评估随机森林模型
# rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
# rf_model.fit(X_train_scaled, y_train)
# y_pred_rf = rf_model.predict(X_test_scaled)
# mse_rf = mean_squared_error(y_test, y_pred_rf)
# r2_rf = r2_score(y_test, y_pred_rf)
#
# print(“模型性能对比(测试集):“)
# print(f“线性回归 — MSE: {mse_lr: .2f}, R²: {r2_lr: .4f}“)
# print(f“随机森林 — MSE: {mse_rf: .2f}, R²: {r2_rf: .4f}“)
# import matplotlib.pyplot as plt
#
# # 获取特征重要性
# importances = rf_model.feature_importances_
# features = X.columns
# indices = np.argsort(importances)[::-1]
#
# # 绘制图表
# plt.figure(figsize=(10, 6))
# plt.title(‘随机森林模型 - 特征重要性‘)
# plt.bar(range(len(importances)), importances[indices], align=‘center‘)
# plt.xticks(range(len(importances)), features[indices], rotation=45)
# plt.tight_layout()
# plt.show()
