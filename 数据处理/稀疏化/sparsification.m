%% 阈值法稀疏化u
% 生成稠密二维数据
rows = 100; % 行数
cols = 100; % 列数
dense_data = rand(rows, cols); % 生成随机的稠密数据

% 显示稠密数据的稀疏性信息
density = nnz(dense_data) / numel(dense_data); % 计算稠密数据的非零元素比例
fprintf('稠密数据的非零元素比例: %.4f\n', density);

% 将稠密数据通过阈值法稀疏化
threshold = 0.5; % 设定阈值，低于此阈值的元素将被视为零
sparse_data_threshold = dense_data;
sparse_data_threshold(abs(sparse_data_threshold) < threshold) = 0; % 根据阈值进行稀疏化处理

% 显示稀疏化后的稀疏性信息
sparse_density_threshold = nnz(sparse_data_threshold) / numel(sparse_data_threshold); % 计算稀疏化数据的非零元素比例
fprintf('阈值法稀疏化后数据的非零元素比例: %.4f\n', sparse_density_threshold);

% 可选：显示原始数据和阈值法稀疏化后的数据
subplot(1, 2, 1);
imagesc(dense_data);
colormap('hot');
colorbar;
title('原始稠密数据');

subplot(1, 2, 2);
imagesc(sparse_data_threshold);
colormap('hot');
colorbar;
title('阈值法稀疏化后数据');
%% 使用稀疏矩阵
% 生成稠密二维数据
rows = 100; % 行数
cols = 100; % 列数
dense_data = rand(rows, cols); % 生成随机的稠密数据

% 将稠密数据转换为稀疏矩阵
sparse_data_sparse = sparse(dense_data);

% 计算稀疏化后的稀疏性信息
sparse_density_sparse = nnz(sparse_data_sparse) / numel(sparse_data_sparse); % 计算稀疏化数据的非零元素比例
fprintf('稀疏矩阵稀疏化后数据的非零元素比例: %.4f\n', sparse_density_sparse);

% 可选：显示原始数据和稀疏矩阵稀疏化后的数据
subplot(1, 2, 1);
imagesc(dense_data);
colormap('hot');
colorbar;
title('原始稠密数据');

subplot(1, 2, 2);
imagesc(full(sparse_data_sparse)); % 显示稀疏矩阵需要使用 full 函数转换为全矩阵
colormap('hot');
colorbar;
title('稀疏矩阵稀疏化后数据');
