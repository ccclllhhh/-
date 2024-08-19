import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_swiss_roll
from sklearn.manifold import LocallyLinearEmbedding
import pandas as pd

# 生成瑞士卷数据
n_samples = 1500
X, color = make_swiss_roll(n_samples)

# 进行局部线性嵌入 (LLE)
n_neighbors = 12
n_components = 2
lle = LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=n_components, method='standard')
X_r = lle.fit_transform(X)

# 绘制原始瑞士卷数据
fig = plt.figure(figsize=(15, 7))

ax = fig.add_subplot(121, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.get_cmap('Spectral'))
ax.set_title("Original Swiss Roll Data")
ax.set_xlabel("X axis")
ax.set_ylabel("Y axis")
ax.set_zlabel("Z axis")

# 绘制降维后的数据
ax = fig.add_subplot(122)
sc = ax.scatter(X_r[:, 0], X_r[:, 1], c=color, cmap=plt.get_cmap('Spectral'))
ax.set_title("2D LLE Projection")
ax.set_xlabel("Component 1")
ax.set_ylabel("Component 2")
plt.colorbar(sc)
plt.show()

# 进一步分析和可视化
# 分析降维后的数据密度分布
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

# Pairplot to see the relationships between the components and the original features
df = pd.DataFrame(X_r, columns=["Component 1", "Component 2"])
df["Color"] = color

sns.pairplot(df, vars=["Component 1", "Component 2"], hue="Color", palette="Spectral")
plt.show()
