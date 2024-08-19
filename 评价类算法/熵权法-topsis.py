
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np

# 设置字体为 SimHei 以支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号


# 熵权法计算权重
def entropy_weight(decision_matrix):
    # 归一化决策矩阵
    scaler = MinMaxScaler()
    normalized_matrix = scaler.fit_transform(decision_matrix)

    # 计算每列的熵值
    m, n = normalized_matrix.shape
    k = 1.0 / np.log(m)
    p = normalized_matrix / normalized_matrix.sum(axis=0)
    e = -k * (p * np.log(p + 1e-10)).sum(axis=0)

    # 计算每列的权重
    d = 1 - e
    weights = d / d.sum()

    return weights


# 示例：构造一个决策矩阵（行是备选方案，列是准则）
decision_matrix = np.array([
    [250, 6, 7],
    [200, 7, 8],
    [300, 8, 5],
])

# 使用熵权法计算权重
weights = entropy_weight(decision_matrix)
print("熵权法计算的权重:", weights)


def topsis(decision_matrix, weights):
    # 归一化决策矩阵
    scaler = MinMaxScaler()
    normalized_matrix = scaler.fit_transform(decision_matrix)

    # 加权归一化决策矩阵
    weighted_matrix = normalized_matrix * weights

    # 确定理想解和负理想解
    ideal_solution = np.max(weighted_matrix, axis=0)
    negative_ideal_solution = np.min(weighted_matrix, axis=0)

    # 计算与理想解和负理想解的距离
    dist_to_ideal = np.sqrt(np.sum((weighted_matrix - ideal_solution) ** 2, axis=1))
    dist_to_negative_ideal = np.sqrt(np.sum((weighted_matrix - negative_ideal_solution) ** 2, axis=1))

    # 计算相对接近度
    closeness = dist_to_negative_ideal / (dist_to_ideal + dist_to_negative_ideal)

    return closeness


# 计算 TOPSIS 得分
scores = topsis(decision_matrix, weights)
print("TOPSIS 得分:", scores)

# 根据得分排序
options = ['方案1', '方案2', '方案3']
rank = np.argsort(scores)[::-1] + 1
print("排序:", rank)

import matplotlib.pyplot as plt

# 绘制柱状图
plt.figure(figsize=(8, 6))
plt.bar(options, scores, color='skyblue')
plt.title('TOPSIS 得分柱状图')
plt.xlabel('方案')
plt.ylabel('TOPSIS 得分')
plt.show()

# 绘制雷达图
normalized_matrix = MinMaxScaler().fit_transform(decision_matrix)
normalized_matrix = np.concatenate((normalized_matrix, normalized_matrix[:,[0]]), axis=1)

labels = ['准则1', '准则2', '准则3']
angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

for i in range(len(options)):
    ax.plot(angles, normalized_matrix[i], linewidth=2, linestyle='solid', label=options[i])
    ax.fill(angles, normalized_matrix[i], alpha=0.25)

ax.set_yticklabels([])
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)

plt.title('各方案的雷达图')
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
plt.show()

