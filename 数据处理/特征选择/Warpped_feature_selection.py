from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn.datasets import make_classification
import numpy as np

# 生成示例数据集
X, y = make_classification(n_samples=100, n_features=20, random_state=0)
# 是 scikit-learn 库中的一个函数，用于生成一个随机的分类问题数据集。
#n_samples=100 表示生成数据集中的样本数量为100。
#n_features=20 表示每个样本具有的特征数量为20。
#random_state=0 是为了确保每次运行代码时生成的数据是一致的，即随机种子固定为0。

# 定义一个SVM分类器作为评估器
svm = SVC(kernel="linear")

# 使用递归特征消除（RFE）作为包裹式特征选择的方法
rfe = RFE(estimator=svm, n_features_to_select=5, step=1)
#estimator=svm：这里指定了用于特征选择的评估器（estimator），即我们希望基于SVM分类器进行特征选择。
#n_features_to_select=5：这个参数指定在特征选择过程中要选择的特征数量。在这个例子中，我们希望最终选择出5个最优的特征。
#step=1：指定每次迭代中移除的特征数目。在这里，每次迭代中移除一个特征，直到达到所需的特征数量为止。

# 将RFE应用于数据集
rfe.fit(X, y)

# 输出选择的特征的排名
print("特征排名：", rfe.ranking_)

# 输出选择的特征
selected_features = [i for i in range(len(rfe.ranking_)) if rfe.support_[i]]
print("选择的特征索引：", selected_features)

# 使用选择的特征子集进行交叉验证评估
selected_X = rfe.transform(X)
scores = cross_val_score(svm, selected_X, y, cv=5)

print("交叉验证分数：", scores)
print("平均交叉验证分数：", scores.mean())
