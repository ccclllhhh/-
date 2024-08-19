import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 生成随机数据
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# 标准化数据
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 添加截距项
X_b = np.c_[np.ones((100, 1)), X_scaled]  # 增加一列1用于截距项

# 初始化参数
m, n = X_b.shape
alpha = 0.1  # 正则化强度
l1_ratio = 0.5  # L1 和 L2 正则化的比例
lambda_l1 = l1_ratio * alpha
lambda_l2 = (1 - l1_ratio) * alpha

theta = np.zeros((n, 1))
z = np.zeros((n, 1))
u = np.zeros((n, 1))

# ADMM 参数
rho = 1.0
max_iter = 1000
tolerance = 1e-4

# ADMM 算法
for iteration in range(max_iter):
    # 更新 theta
    A = X_b.T @ X_b + rho * np.eye(n)
    b = X_b.T @ y + rho * (z - u)
    theta = np.linalg.solve(A, b)

    # 更新 z（软阈值化）
    z_old = z.copy()
    z = np.sign(theta + u) * np.maximum(np.abs(theta + u) - lambda_l1 / rho, 0)

    # 更新 u
    u += theta - z

    # 计算收敛性
    primal_residual = np.linalg.norm(theta - z)
    dual_residual = np.linalg.norm(-rho * (z - z_old))
    if primal_residual < tolerance and dual_residual < tolerance:
        break

# 预测
y_pred = X_b.dot(theta)

# 绘图
plt.scatter(X, y, color='blue', label='Data points')
plt.plot(X, y_pred, color='red', label='Elastic Net regression line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Elastic Net Regression with ADMM')
plt.legend()
plt.show()

print("Optimized parameters:", theta.ravel())
