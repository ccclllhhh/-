#使用DBSCAN算法的一个详细案例，包括数据分析和可视化。
# 我们将使用Scikit-learn库来实现DBSCAN，并生成两个以上的数据可视化图形。
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

# 生成合成数据
n_samples = 1500
centers = [[1, 1], [-1, -1], [1, -1]]
X, _ = make_blobs(n_samples=n_samples, centers=centers, cluster_std=0.4, random_state=0)

# 数据标准化
X = StandardScaler().fit_transform(X)

plt.scatter(X[:, 0], X[:, 1], s=10)
plt.title("Generated Data")
plt.show()

#使用DBSCAN算法对生成的数据进行聚类。
from sklearn.cluster import DBSCAN

# 应用DBSCAN算法
db = DBSCAN(eps=0.3, min_samples=10).fit(X)
labels = db.labels_

# 获取核心样本的掩码
core_samples_mask = np.zeros_like(labels, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True

# 标记噪声
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print(f'Estimated number of clusters: {n_clusters_}')
print(f'Estimated number of noise points: {n_noise_}')

#聚类结果可视化
# 黑色用于标记噪声
unique_labels = set(labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

plt.figure(figsize=(10, 7))
for k, col in zip(unique_labels, colors):
    if k == -1:
        # 黑色用于标记噪声
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=14)

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

plt.title('DBSCAN Clustering Results')
plt.show()

#我们可以尝试不同的eps值来观察其对聚类结果的影响。
eps_values = [0.1, 0.3, 0.5, 0.7]
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.flatten()

for ax, eps in zip(axes, eps_values):
    db = DBSCAN(eps=eps, min_samples=10).fit(X)
    labels = db.labels_

    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = X[class_member_mask & core_samples_mask]
        ax.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                markeredgecolor='k', markersize=14)

        xy = X[class_member_mask & ~core_samples_mask]
        ax.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                markeredgecolor='k', markersize=6)

    ax.set_title(f'DBSCAN with eps={eps}')

plt.tight_layout()
plt.show()