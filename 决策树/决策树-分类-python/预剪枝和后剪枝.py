import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report

# 加载鸢尾花数据集
iris = datasets.load_iris()
X = iris.data[:, :2]  # 为了便于可视化，只选择前两个特征
y = iris.target

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 预剪枝：限制树的深度
pre_pruned_model = DecisionTreeClassifier(max_depth=3)
pre_pruned_model.fit(X_train, y_train)

# 绘制预剪枝后的决策树
plt.figure(figsize=(10, 8))
plot_tree(pre_pruned_model, filled=True, feature_names=iris.feature_names[:2], class_names=iris.target_names)
plt.title("Pre-pruned Decision Tree")
plt.show()

# 后剪枝：使用 GridSearchCV 寻找最佳的剪枝参数
params = {'ccp_alpha': np.arange(0, 0.05, 0.01)}
tree = DecisionTreeClassifier(random_state=42)
grid_search = GridSearchCV(tree, params, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# 输出最佳参数和得分
best_alpha = grid_search.best_params_['ccp_alpha']
print(f"Best alpha: {best_alpha}")

# 使用最佳的剪枝参数训练决策树
post_pruned_model = DecisionTreeClassifier(ccp_alpha=best_alpha, random_state=42)
post_pruned_model.fit(X_train, y_train)

# 绘制后剪枝后的决策树
plt.figure(figsize=(10, 8))
plot_tree(post_pruned_model, filled=True, feature_names=iris.feature_names[:2], class_names=iris.target_names)
plt.title("Post-pruned Decision Tree")
plt.show()

# 评估模型性能
y_pred = post_pruned_model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=iris.target_names))
