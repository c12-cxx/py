import numpy as np
import matplotlib.pyplot as plt

# 数据
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

# 初始化 w 和 b 范围
W = np.arange(0.0, 4.0, 0.1)  # 权重 w 的取值范围
B = np.arange(0.0, 4.0, 0.1)  # 偏置 b 的取值范围


# 计算预测值
def forward(x):
    return x * w + b


# 计算损失
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2


# 用于存储每次迭代的损失
w_list = []
b_list = []
mse_list = []

# 双层循环遍历 w 和 b 的不同值
for w in W:
    for b in B:
        l_sum = 0
        for x_val, y_val in zip(x_data, y_data):
            y_pred_val = forward(x_val)
            loss_val = loss(x_val, y_val)
            l_sum += loss_val
        mse = l_sum / len(x_data)  # 平均损失
        w_list.append(w)
        b_list.append(b)
        mse_list.append(mse)

# 可视化损失曲面
w_grid, b_grid = np.meshgrid(W, B)  # 创建 w 和 b 的网格
mse_array = np.array(mse_list).reshape(len(W), len(B))  # 将 mse_list 转换成二维数组

# 画图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(w_grid, b_grid, mse_array, cmap='viridis')

# 添加标签
ax.set_xlabel('w')
ax.set_ylabel('b')
ax.set_zlabel('MSE')
plt.title('Loss Surface for Different w and b')
plt.show()
