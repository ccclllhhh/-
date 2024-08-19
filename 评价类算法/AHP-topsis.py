
import matplotlib.pyplot as plt
import numpy as np

# 设置字体为 SimHei 以支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

def ahp_weight(matrix):
    eig_vals, eig_vecs = np.linalg.eig(matrix)
    max_eig_val = np.max(eig_vals)
    max_eig_vec = eig_vecs[:, np.argmax(eig_vals)]
    weights = max_eig_vec / np.sum(max_eig_vec)
    return np.real(weights)

# 示例：构造一个判断矩阵
matrix = np.array([
    [1, 1/2, 3],
    [2, 1, 4],
    [1/3, 1/4, 1]
])

# 计算权重
weights = ahp_weight(matrix)
print("权重:", weights)

from sklearn.preprocessing import MinMaxScaler


# TOPSIS 方法实现
def topsis(decision_matrix, weights):
    # Step 1: 归一化决策矩阵
    scaler = MinMaxScaler()
    normalized_matrix = scaler.fit_transform(decision_matrix)

    # Step 2: 加权归一化决策矩阵
    weighted_matrix = normalized_matrix * weights

    # Step 3: 确定理想解和负理想解
    ideal_solution = np.max(weighted_matrix, axis=0)
    negative_ideal_solution = np.min(weighted_matrix, axis=0)

    # Step 4: 计算与理想解和负理想解的距离
    dist_to_ideal = np.sqrt(np.sum((weighted_matrix - ideal_solution) ** 2, axis=1))
    dist_to_negative_ideal = np.sqrt(np.sum((weighted_matrix - negative_ideal_solution) ** 2, axis=1))

    # Step 5: 计算相对接近度
    closeness = dist_to_negative_ideal / (dist_to_ideal + dist_to_negative_ideal)

    return closeness


# 示例：构造一个决策矩阵（行是备选方案，列是准则）
decision_matrix = np.array([
    [250, 6, 7],
    [200, 7, 8],
    [300, 8, 5],
])

# 使用 AHP 计算的权重
weights = np.array([0.4, 0.35, 0.25])

# 计算 TOPSIS 得分
scores = topsis(decision_matrix, weights)
print("TOPSIS 得分:", scores)

# 根据得分排序
rank = np.argsort(scores)[::-1] + 1
print("排序:", rank)

# topsis 得分柱状图
import matplotlib.pyplot as plt
import numpy as np

# 定义方案的名称
options = ['方案1', '方案2', '方案3']

# 绘制柱状图
plt.figure(figsize=(8, 6))
plt.bar(options, scores, color='skyblue')
plt.title('TOPSIS 得分柱状图')
plt.xlabel('方案')
plt.ylabel('TOPSIS 得分')
plt.show()

#各方案雷达图
import matplotlib.pyplot as plt
import numpy as np

# 归一化处理的决策矩阵
normalized_matrix = MinMaxScaler().fit_transform(decision_matrix)

# 添加一个列用于闭合雷达图
normalized_matrix = np.concatenate((normalized_matrix, normalized_matrix[:,[0]]), axis=1)

# 定义角度和准则名称
labels = ['准则1', '准则2', '准则3']
angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
angles += angles[:1]

# 创建雷达图
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

# 绘制每个方案的雷达图
for i in range(len(options)):
    ax.plot(angles, normalized_matrix[i], linewidth=2, linestyle='solid', label=options[i])
    ax.fill(angles, normalized_matrix[i], alpha=0.25)

# 添加特征
ax.set_yticklabels([])
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)

plt.title('各方案的雷达图')
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
plt.show()
