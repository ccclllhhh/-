% 生成随机数据
rng(1); % 设置随机数种子以确保结果可重复

% 数据维度为三维，共生成100个样本
X = randn(100, 3); 
y = randi([0 1], 100, 1); % 随机生成0和1的分类标签

% 构建决策树模型
tree = fitctree(X, y, 'MaxNumSplits', 10); % 最大深度为5

% 绘制决策树
view(tree, 'Mode', 'Graph');

% 预测新数据并显示部分结果
Xtest = randn(5, 3); % 生成5个新样本
ytest_pred = predict(tree, Xtest); % 预测分类标签

disp('预测结果:');
disp([Xtest ytest_pred]);

% 计算模型精度（可选）
ytest_true = randi([0 1], 5, 1); % 为新数据生成真实的分类标签
accuracy = sum(ytest_pred == ytest_true) / numel(ytest_true);
disp(['模型精度: ' num2str(accuracy)]);

%% 后剪枝
rng(1); % 设置随机数种子以确保结果可重复

% 生成随机数据
X = randn(100, 3); % 数据维度为三维，共生成100个样本
y = randi([0 1], 100, 1); % 随机生成0和1的分类标签

% 构建决策树模型并进行剪枝（两层）
tree = fitctree(X, y, 'MaxNumSplits', 10, 'Prune', 'on'); % 最大深度为5并启用剪枝

% 绘制决策树
view(tree, 'Mode', 'Graph');

% 预测新数据并显示部分结果
Xtest = randn(5, 3); % 生成5个新样本
ytest_pred = predict(tree, Xtest); % 预测分类标签

disp('预测结果:');
disp([Xtest ytest_pred]);

% 计算模型精度
ytest_true = randi([0 1], 5, 1); % 为新数据生成真实的分类标签
accuracy = sum(ytest_pred == ytest_true) / numel(ytest_true);
disp(['模型精度: ' num2str(accuracy)]);



