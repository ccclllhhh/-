import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.datasets import make_blobs

# 生成模拟数据
n_samples = 1500
random_state = 170
X, y = make_blobs(n_samples=n_samples, random_state=random_state)

# 层次聚类
Z = linkage(X, 'ward')

# 绘制层次聚类树状图
plt.figure(figsize=(10, 7))
plt.title("Dendrogram")
dendrogram(Z)
plt.xlabel('Sample index')
plt.ylabel('Distance')
plt.show()

# 聚类数目设置为3，获取聚类标签
k = 3
clusters = fcluster(Z, k, criterion='maxclust')

# 创建DataFrame以便于后续处理和可视化
df = pd.DataFrame(X, columns=['Feature1', 'Feature2'])
df['Cluster'] = clusters

# 绘制聚类结果的散点图
plt.figure(figsize=(10, 7))
sns.scatterplot(data=df, x='Feature1', y='Feature2', hue='Cluster', palette='viridis')
plt.title('Scatter Plot of Clusters')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

# 计算每个聚类的质心
centroids = df.groupby('Cluster').mean().reset_index()

# 绘制聚类结果及质心的散点图
plt.figure(figsize=(10, 7))
sns.scatterplot(data=df, x='Feature1', y='Feature2', hue='Cluster', palette='viridis', legend='full')
plt.scatter(centroids['Feature1'], centroids['Feature2'], s=300, c='red', marker='X')
for i, centroid in centroids.iterrows():
    plt.text(centroid['Feature1'], centroid['Feature2'], f'Centroid {int(centroid["Cluster"])}', fontsize=12, weight='bold', color='red')
plt.title('Clusters with Centroids')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

# 使用Seaborn绘制带有KDE的对角线散点图矩阵
sns.pairplot(df, hue='Cluster', palette='viridis', diag_kind='kde', markers=['o', 's', 'D'])
plt.suptitle('Pairplot with KDE Diagonals')
plt.show()