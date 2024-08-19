% 生成随机数据
rng(1); % 设置随机数种子以确保结果可重复

% 数据维度为三维，共生成100个样本
X = randn(100, 3); 
y = randn(100, 1); % 随机生成连续的目标变量

% 构建决策回归树模型
tree = fitrtree(X, y, 'MaxNumSplits', 5); % 最大深度为4

% 绘制决策树
view(tree, 'Mode', 'Graph');

% 预测新数据并显示部分结果
Xtest = randn(5, 3); % 生成5个新样本
ytest_pred = predict(tree, Xtest); % 预测目标变量

disp('预测结果:');
disp([Xtest ytest_pred]);

% 计算预测误差（可选）
ytest_true = randn(5, 1); % 为新数据生成真实的目标变量
mse = mean((ytest_pred - ytest_true).^2);
disp(['预测均方误差: ' num2str(mse)]);


%% 后剪枝操作
rng(1); % 设置随机数种子以确保结果可重复

% 数据维度为三维，共生成100个样本
X = randn(100, 3); 
y = randn(100, 1); % 随机生成连续的目标变量

% 构建决策回归树模型并进行剪枝
tree = fitrtree(X, y, 'MaxNumSplits', 5, 'Prune', 'on', 'MinParentSize', 10, 'MinLeafSize', 5);
%它指定了一个节点必须具有的最小样本数，才能够继续分裂成子节点。如果节点的样本数少于
% MinParentSize，则该节点不会再分裂，成为叶节点（叶子）。
% MinLeafSize:
%它指定了叶节点（叶子）必须具有的最小样本数。
% 如果在进行分裂时得到的叶节点样本数少于 MinLeafSize，则该分裂不会发生，节点保持为叶节点。

% 绘制决策树
view(tree, 'Mode', 'Graph');

% 预测新数据并显示部分结果
Xtest = randn(5, 3); % 生成5个新样本
ytest_pred = predict(tree, Xtest); % 预测目标变量

disp('预测结果:');
disp([Xtest ytest_pred]);

% 计算预测误差（可选）
ytest_true = randn(5, 1); % 为新数据生成真实的目标变量
mse = mean((ytest_pred - ytest_true).^2);
disp(['预测均方误差: ' num2str(mse)]);



