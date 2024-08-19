import numpy as np
from sklearn.feature_selection import mutual_info_regression

# 生成数据
np.random.seed(0)
n = 1000  # 样本数量
num_features = 4  # 特征数量

# 假设有4个特征和1个目标变量
X = np.random.randn(n, num_features)  # 生成服从正态分布的特征数据
Y = np.sum(X, axis=1) + np.random.randn(n)  # 目标变量是特征之和再加上一些随机噪声

# 计算互信息
MI = mutual_info_regression(X, Y)

# 显示计算结果
print("各特征与目标变量的互信息：")
for i in range(num_features):
    print(f"特征 {i+1}: {MI[i]}")
#挑选最高的