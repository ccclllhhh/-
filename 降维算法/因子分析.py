import pandas as pd
import numpy as np

# 创建虚拟数据集
np.random.seed(0)
n_samples = 1000
n_features = 5

# 生成数据集
data = np.random.randint(1, 11, size=(n_samples, n_features))
df = pd.DataFrame(data, columns=['Screen Size', 'Camera Quality', 'Battery Life', 'Performance', 'Design'])

# 添加一些噪音列
df['other factory'] = np.random.randint(1, 11, size=n_samples)

# 显示数据集的前几行
print(df.head())



from sklearn.decomposition import FactorAnalysis
import matplotlib.pyplot as plt

# 因子分析
fa = FactorAnalysis(n_components=3)
fa.fit(df)

# 提取因子负荷
factors = pd.DataFrame(fa.components_, columns=df.columns)

# 绘制因子负荷热图
plt.figure(figsize=(10, 6))
plt.imshow(factors, cmap='viridis', aspect='auto', interpolation='nearest')
plt.colorbar()
plt.xticks(range(len(df.columns)), df.columns, rotation=45)
plt.yticks(range(len(factors)), ['Factor {}'.format(i+1) for i in range(len(factors))])
plt.title('Factor Loading Heatmap')
plt.show()

# 绘制因子得分散点图
factors_scores = fa.transform(df)
plt.figure(figsize=(8, 6))
plt.scatter(factors_scores[:, 0], factors_scores[:, 1])
plt.xlabel('Factor 1')
plt.ylabel('Factor 2')
plt.title('Factor Scores Scatter Plot')
plt.grid(True)
plt.show()






