x_data = [1, 2, 3]

y_data = [2, 4, 6]

w = 4.0


def forward(x):
    return x * w


def cost(xs, ys):
    costvalue = 0
    for x, y in zip(xs, ys):
        y_pred = forward(x)
        costvalue += (y_pred - y) ** 2
    return costvalue / len(xs)


def gradient(xs, ys):
    grad = 0
    for x, y in zip(xs, ys):
        grad += 2 * x * (w * x - y)
    return grad / len(xs)


for epoch in range(199):
    cost_val = cost(x_data, y_data)

    grad_val = gradient(x_data, y_data)

    w = w - 0.01 * grad_val

    print('训练轮次', epoch, 'w=', w, 'loss', cost_val)

print("此时w已经训练好了，用训练好的w进行模型推理", forward(5))
