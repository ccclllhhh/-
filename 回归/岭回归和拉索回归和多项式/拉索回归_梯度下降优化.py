import numpy as np
import matplotlib.pyplot as plt

# 生成随机数据
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# 添加截距项
X_b = np.c_[np.ones((100, 1)), X]  # 增加一列1用于截距项

# 初始化参数
theta = np.random.randn(2, 1)  # 两个参数，包括截距和斜率
alpha = 0.1  # 学习率
iterations = 1000
lambda_reg = 0.1  # 拉索回归的正则化参数

# 梯度下降
m = len(y)
for iteration in range(iterations):
    predictions = X_b.dot(theta)
    errors = predictions - y
    gradient = (2/m) * X_b.T.dot(errors) + lambda_reg * np.sign(theta)
    theta = theta - alpha * gradient

# 预测
y_pred = X_b.dot(theta)

# 绘图
plt.scatter(X, y, color='blue', label='Data points')
plt.plot(X, y_pred, color='red', label='Lasso regression line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Lasso Regression with Gradient Descent')
plt.legend()
plt.show()

print("Optimized parameters:", theta.ravel())
