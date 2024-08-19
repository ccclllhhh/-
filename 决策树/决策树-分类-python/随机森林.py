import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor

X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)

# 训练随机森林回归模型
regressor = RandomForestRegressor(n_estimators=100, random_state=42)
regressor.fit(X, y)

# 预测结果
X_test = np.linspace(-3, 3, 100).reshape(-1, 1)
y_pred = regressor.predict(X_test)

# 绘制随机森林中的一棵决策树图像
estimator = regressor.estimators_[0]

plt.figure(figsize=(10, 8))
plt.scatter(X, y, color="b", s=30, marker="o", label="training data")
plt.plot(X_test, y_pred, color="r", label="predictions")
plt.title("Random Forest Regression")
plt.xlabel("Feature")
plt.ylabel("Target")
plt.legend(loc="upper left")

# 可视化一棵决策树
plt.figure(figsize=(15, 10))
from sklearn.tree import plot_tree
plot_tree(estimator, filled=True, feature_names=['Feature'])
plt.title("Example Decision Tree from Random Forest")
plt.show()