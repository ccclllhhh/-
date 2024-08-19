import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_swiss_roll
from sklearn.manifold import LocallyLinearEmbedding
import pandas as pd

# 生成瑞士卷数据
n_samples = 1500
X, color = make_swiss_roll(n_samples)

# 进行局部线性嵌入 (LLE) 降维到三维
n_neighbors = 12
n_components = 3
lle = LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=n_components, method='standard')
X_r = lle.fit_transform(X)

# 绘制原始瑞士卷数据
fig = plt.figure(figsize=(18, 7))

# 原始瑞士卷数据
ax = fig.add_subplot(131, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.get_cmap('Spectral'))
ax.set_title("Original Swiss Roll Data")
ax.set_xlabel("X axis")
ax.set_ylabel("Y axis")
ax.set_zlabel("Z axis")

# 降维后的三维数据
ax = fig.add_subplot(132, projection='3d')
ax.scatter(X_r[:, 0], X_r[:, 1], X_r[:, 2], c=color, cmap=plt.get_cmap('Spectral'))
ax.set_title("3D LLE Projection")
ax.set_xlabel("Component 1")
ax.set_ylabel("Component 2")
ax.set_zlabel("Component 3")

plt.show()

# 进一步分析和可视化
# 降维后的数据密度分布（只选择前两个主成分进行二维密度图绘制）
sns.set(style="whitegrid")
plt.figure(figsize=(12, 6))

# KDE plot of the first component
plt.subplot(1, 2, 1)
sns.kdeplot(X_r[:, 0], fill=True, color="r")
plt.title("Density plot of Component 1")

# KDE plot of the second component
plt.subplot(1, 2, 2)
sns.kdeplot(X_r[:, 1], fill=True, color="b")
plt.title("Density plot of Component 2")

plt.show()

# 对于三维数据，pairplot 不适用，改为二维散点图查看前两个主成分的关系
df = pd.DataFrame(X_r[:, :2], columns=["Component 1", "Component 2"])
df["Color"] = color

sns.pairplot(df, vars=["Component 1", "Component 2"], hue="Color", palette="Spectral")
plt.show()
