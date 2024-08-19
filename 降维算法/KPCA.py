import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_circles

# 生成非线性数据
X, y = make_circles(n_samples=800, factor=0.3, noise=0.1, random_state=42)

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 核主成分分析
kpca = KernelPCA(n_components=2, kernel='rbf', gamma=15)
X_kpca = kpca.fit_transform(X_scaled)

# 可视化原始数据
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap=plt.cm.Spectral)
plt.title('Original Data')

# 可视化核主成分分析后的数据
plt.subplot(1, 2, 2)
plt.scatter(X_kpca[:, 0], X_kpca[:, 1], c=y, cmap=plt.cm.Spectral)
plt.title('Kernel PCA Transformed Data')
plt.show()

# 额外图形：核PCA前后的特征分布
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.hist(X_scaled[:, 0], bins=30, alpha=0.5, label='Feature 1')
ax1.hist(X_scaled[:, 1], bins=30, alpha=0.5, label='Feature 2')
ax1.set_title('Original Features Distribution')
ax1.legend()

ax2.hist(X_kpca[:, 0], bins=30, alpha=0.5, label='Principal Component 1')
ax2.hist(X_kpca[:, 1], bins=30, alpha=0.5, label='Principal Component 2')
ax2.set_title('Transformed Principal Components Distribution')
ax2.legend()

plt.show()