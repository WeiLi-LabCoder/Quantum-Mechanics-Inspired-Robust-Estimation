clear; clc;  

% 读取 fig
fig = openfig('com_NMSE_X_HuberLAD_all.fig', 'invisible');

% 找到坐标轴
ax = findobj(fig, 'Type', 'axes');

% 找到所有曲线对象
lines = findobj(ax, 'Type', 'line');

% 为了保持和图中显示顺序一致，翻转一下
lines = flipud(lines);

% 提取数据
data = struct('name', {}, 'x', {}, 'y', {}, 'color', {}, ...
              'linestyle', {}, 'marker', {});

for k = 1:length(lines)
    data(k).name      = get(lines(k), 'DisplayName');
    data(k).x         = get(lines(k), 'XData');
    data(k).y         = get(lines(k), 'YData');
    data(k).color     = get(lines(k), 'Color');
    data(k).linestyle = get(lines(k), 'LineStyle');
    data(k).marker    = get(lines(k), 'Marker');
end

% 显示提取结果
for k = 1:length(data)
    fprintf('%2d: %s\n', k, data(k).name);
end

% 可选：保存成 mat 文件
save('extracted_curves.mat', 'data');