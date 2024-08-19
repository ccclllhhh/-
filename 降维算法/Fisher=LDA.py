import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei']  # 黑体
rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示为方块的问题

# 加载数据集
data = load_iris()
X = data.data
y = data.target
feature_names = data.feature_names
target_names = data.target_names

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 切分数据集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Fisher判别分析（FDA）
fda = LinearDiscriminantAnalysis(n_components=2)  # 降到2维
X_train_fda = fda.fit_transform(X_train, y_train)
X_test_fda = fda.transform(X_test)

# 打印Fisher判别向量
print("Fisher判别向量（系数）：")
print(fda.coef_)

# 进行分类
clf = LogisticRegression()
clf.fit(X_train_fda, y_train)
y_pred = clf.predict(X_test_fda)

# 评估分类器
print("准确率:", accuracy_score(y_test, y_pred))
print("分类报告:")
print(classification_report(y_test, y_pred, target_names=target_names))
print("混淆矩阵:")
print(confusion_matrix(y_test, y_pred))

# 可视化FDA结果
plt.figure(figsize=(10, 6))
colors = ['navy', 'turquoise', 'darkorange']
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_test_fda[y_test == i, 0], X_test_fda[y_test == i, 1], color=color, lw=2, label=target_name)
plt.xlabel('FDA 组件 1')
plt.ylabel('FDA 组件 2')
plt.title('Fisher判别分析')
plt.legend(loc='best')
plt.grid()
plt.show()
