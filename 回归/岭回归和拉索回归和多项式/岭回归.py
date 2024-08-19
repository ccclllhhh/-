import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 生成随机数据
np.random.seed(0)
X = 2 * np.random.rand(100, 2)
y = 4 + 3 * X[:, [0]] + 2 * X[:, [1]] + np.random.randn(100, 1)

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 切分数据集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=0)

# 岭回归模型
ridge_reg = Ridge(alpha=1, solver="cholesky")
ridge_reg.fit(X_train, y_train)

# 生成网格数据用于绘制回归平面
x0, x1 = np.meshgrid(
    np.linspace(X_scaled[:, 0].min(), X_scaled[:, 0].max(), 100),
    np.linspace(X_scaled[:, 1].min(), X_scaled[:, 1].max(), 100)
)
x0x1 = np.c_[x0.ravel(), x1.ravel()]
y_pred = ridge_reg.predict(x0x1).reshape(x0.shape)

# 绘图
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# 绘制训练数据点
ax.scatter(X_train[:, 0], X_train[:, 1], y_train, color='blue', label='Train data points')

# 绘制回归平面
ax.plot_surface(x0, x1, y_pred, color='red', alpha=0.5, label='Ridge regression plane')

ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Target')
ax.set_title('Ridge Regression with Multiple Features')

plt.show()
