% 生成随机数据
rng(1); % 设置随机数种子以保证结果的可重复性
num_points = 1000;
data = [randn(num_points, 2) * 0.75 + 1.5;
        randn(num_points, 2) * 0.5 - 1.5;
        randn(num_points, 2) * 0.5 + [-2, 2]];

% 执行 kmeans 聚类
k = 3; % 聚类个数
[idx, C] = kmeans(data, k);

% 绘制聚类结果
figure;
gscatter(data(:,1), data(:,2), idx, 'rgb');
hold on;
plot(C(:,1), C(:,2), 'kx', 'MarkerSize', 15, 'LineWidth', 3);
title(['K-means 聚类结果 (K = ', num2str(k), ')']);
legend('Cluster 1', 'Cluster 2', 'Cluster 3', 'Centroids', 'Location', 'Best');
hold off;