% 原始数据
data = [12, 35, 27, 20, 50, 45, 30, 15, 5];

% 排序数据
sorted_data = sort(data);

% 确定区间个数
num_bins = 3;

% 计算区间边界
bin_boundaries = linspace(min(sorted_data), max(sorted_data), num_bins+1);

% 离散化数据
discretized_data = discretize(sorted_data, bin_boundaries);

% 输出结果
disp('原始数据:');
disp(data);
disp('排序后的数据:');
disp(sorted_data);
disp(['离散化后的数据 (分为', num2str(num_bins), '个区间):']);
disp(discretized_data);
%% 等频率
% 原始数据
data = [12, 35, 27, 20, 50, 45, 30, 15, 5];

% 排序数据
sorted_data = sort(data);

% 确定区间个数
num_bins = 3;

% 计算每个区间的数据点数量
bin_size = ceil(length(sorted_data) / num_bins);

% 确定区间边界
bin_boundaries = zeros(1, num_bins+1);
bin_boundaries(1) = sorted_data(1);
bin_boundaries(end) = sorted_data(end);

for i = 1:num_bins-1
    bin_boundaries(i+1) = sorted_data(i * bin_size);
end

% 离散化数据
discretized_data = discretize(sorted_data, bin_boundaries);

% 输出结果
disp('原始数据:');
disp(data);
disp('排序后的数据:');
disp(sorted_data);
disp(['等频率离散化后的数据 (分为', num2str(num_bins), '个区间):']);
disp(discretized_data);
