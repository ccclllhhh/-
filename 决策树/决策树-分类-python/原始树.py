import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier, plot_tree

# 加载鸢尾花数据集
iris = datasets.load_iris()
X = iris.data[:, :2]  # 为了便于可视化，只选择前两个特征
y = iris.target

# 拟合决策树模型
model = DecisionTreeClassifier()
model.fit(X, y)

# 绘制决策树
plt.figure(figsize=(10, 8))
plot_tree(model, filled=True, feature_names=iris.feature_names[:2], class_names=iris.target_names)
plt.show()
