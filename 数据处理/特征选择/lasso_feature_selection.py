import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 生成一个随机的回归数据集
X, y = make_regression(n_samples=100, n_features=20, noise=0.1, random_state=42)

# 对数据进行标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 定义 Lasso 模型
lasso = Lasso(alpha=0.1)  # alpha 是 L1 正则化的强度参数

# 拟合模型
lasso.fit(X_scaled, y)

# 输出各个特征的系数
print("Coefficients:", lasso.coef_)

# 可视化各个特征的系数
plt.figure(figsize=(10, 6))
plt.plot(range(len(lasso.coef_)), lasso.coef_, marker='o', linestyle='None', markersize=8)
plt.xticks(range(len(lasso.coef_)), range(1, len(lasso.coef_)+1))
plt.xlabel('Feature')
plt.ylabel('Coefficient')
plt.title('Lasso Coefficients')
plt.grid(True)
plt.show()

# 可以进一步使用交叉验证来选择最优的 alpha 参数
alphas = np.logspace(-4, 0, 50)
lasso_cv = LassoCV(alphas=alphas, cv=5)
lasso_cv.fit(X_scaled, y)

print("Best alpha using built-in LassoCV:", lasso_cv.alpha_)

# 使用最优 alpha 再次拟合模型
lasso_best = Lasso(alpha=lasso_cv.alpha_)
lasso_best.fit(X_scaled, y)

# 输出最优模型的系数
print("Coefficients (Best Lasso):", lasso_best.coef_)
