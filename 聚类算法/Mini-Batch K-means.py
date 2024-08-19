import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import MiniBatchKMeans

# 生成合成数据集
X, y = make_blobs(n_samples=1500, centers=3, cluster_std=1.0, random_state=42)

# 使用Mini-Batch K-means进行聚类
kmeans = MiniBatchKMeans(n_clusters=3, batch_size=100, random_state=42)
kmeans.fit(X)

# 获取聚类中心和预测标签
centers = kmeans.cluster_centers_
labels = kmeans.labels_

# 绘制原始数据的散点图
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap='viridis', label='Original Data')
plt.colorbar()
plt.title('Original Data with True Labels')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

# 绘制Mini-Batch K-means聚类结果的散点图
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, s=30, cmap='viridis', label='Clustered Data')
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, label='Cluster Centers')
plt.colorbar()
plt.title('Mini-Batch K-means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

# 绘制决策边界
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

Z = Z.reshape(xx.shape)
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, cmap='viridis', alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=30, cmap='viridis')
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, label='Cluster Centers')
plt.colorbar()
plt.title('Mini-Batch K-means Decision Boundaries')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()