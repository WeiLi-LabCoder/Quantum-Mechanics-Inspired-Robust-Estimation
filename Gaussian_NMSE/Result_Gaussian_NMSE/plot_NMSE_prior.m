 
clear; clc; 

%% ===== 1. 读取两个 fig 文件 =====
figFiles = { ...
    'com_NMSE_X_prior.fig' 
    };

nScenario = numel(figFiles);
dataAll = struct('name',[],'x',[],'y',[], ...
                 'alg',[],'priorType',[],'scenario',[]);

idx = 0;
for s = 1:nScenario
    hFig = openfig(figFiles{s}, 'invisible');
    hAx   = findobj(hFig, 'Type', 'axes');
    hLine = findobj(hAx, 'Type', 'line');
    hLine = flipud(hLine);   % 翻转顺序以匹配 legend

    for k = 1:numel(hLine)
        idx = idx + 1;
        dataAll(idx).name     = string(hLine(k).DisplayName);
        dataAll(idx).x        = hLine(k).XData;
        dataAll(idx).y        = hLine(k).YData;
        dataAll(idx).scenario = s;
    end
    close(hFig);
end
nLine = numel(dataAll);

%% ===== 2. 解析算法与 prior 类型 =====
for k = 1:nLine
    nm = lower(dataAll(k).name);

    if contains(nm, "partial prior")
        dataAll(k).priorType = 2;
    elseif contains(nm, "no prior")
        dataAll(k).priorType = 3;
    else
        dataAll(k).priorType = 1;
    end

    if contains(nm, "pnp-vamp-qd")
        dataAll(k).alg = 3;
    elseif contains(nm, "qd-vamp")
        dataAll(k).alg = 2;
    else
        dataAll(k).alg = 1;
    end
end

%% ===== 3. 全局 SNR / 误差范围 =====
allSNR = dataAll(1).x;
allErr = vertcat(dataAll.y);

minSNR = min(allSNR);
maxSNR = max(allSNR);
minErr = min(allErr(:));
maxErr = max(allErr(:));

zmin = 10^floor(log10(minErr));
zmax = 10^ceil(log10(maxErr));

%% ===== 4. 一列两行布局 =====
set(groot,'defaultAxesTickLabelInterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');
set(groot,'defaultTextInterpreter','latex');

figure  


    ax = gca;
    hold(ax,'on'); box(ax,'on'); grid(ax,'off');

    plotScenarioPanel(ax, dataAll, s, allSNR, ...
                      minSNR, maxSNR, zmin, zmax);
 
exportgraphics(gcf, 'fig_NMSE_X_prior.pdf', ...
    'ContentType','vector', ...
    'BackgroundColor','white'); 

%% ===== 子函数：绘制单个三面板 =====
function plotScenarioPanel(ax, dataAll, scenarioIdx, allSNR, ...
                           minSNR, maxSNR, zmin, zmax)
hold(ax,'on');

xPanel = [-1 0 1];

% 配色
colorList = [
    0.36 0.64 0.60;   % VAMP
    0.25 0.47 0.63;   % QD-VAMP
    0.73 0.24 0.15;   % PnP-VAMP-QD
];
markerList = {'o','s','d'};

panelColors = [
    0.92 0.95 0.88;   % Morandi green
    0.85 0.92 0.94;   % Morandi blue
    0.94 0.83 0.76;   % Morandi pink
];

%% --- 背景面板 + 数据点对齐网格 ---
for pt = 1:3
    x_plane = xPanel(pt);
    y_ticks = allSNR;
    z_ticks = logspace(log10(zmin), log10(zmax), numel(y_ticks));
    [Y, Z] = meshgrid(y_ticks, z_ticks);

    % 面板背景
    surf(ax, x_plane*ones(size(Y)), Y, Z, ...
        'FaceColor',panelColors(pt,:), ...
        'FaceAlpha',0.35, ...
        'EdgeColor','none');

    % 数据点对齐的浅网格
    gridColor = [0.82 0.82 0.82];
    gridWidth = 0.3;

    % 横线
    for i = 1:numel(z_ticks)
        plot3(ax, x_plane*ones(size(y_ticks)), ...
            y_ticks, z_ticks(i)*ones(size(y_ticks)), ...
            'Color',gridColor,'LineWidth',gridWidth);
    end
    % 竖线
    for j = 1:numel(y_ticks)
        plot3(ax, x_plane*ones(size(z_ticks)), ...
            y_ticks(j)*ones(size(z_ticks)), z_ticks, ...
            'Color',gridColor,'LineWidth',gridWidth);
    end
end

%% --- 画曲线 ---
hLeg = gobjects(3,1);
usedAlg = false(3,1);

for k = 1:numel(dataAll)
    if dataAll(k).scenario ~= scenarioIdx
        continue;
    end
    alg = dataAll(k).alg;
    pt  = dataAll(k).priorType;

    x = repmat(xPanel(pt), size(dataAll(k).x));
    y = dataAll(k).x;
    z = dataAll(k).y;

    h = plot3(ax, x, y, z, ...
        'LineWidth',1.5, ...
        'Color',colorList(alg,:), ...
        'Marker',markerList{alg}, ...
        'MarkerSize',5, ...
        'MarkerFaceColor','w');

    if ~usedAlg(alg)
        hLeg(alg) = h;
        usedAlg(alg) = true;
    end
end

%% --- 坐标轴与文字 ---
set(ax,'ZScale','log');
zlim(ax,[zmin zmax]);
ylim(ax,[minSNR maxSNR]);
set(ax,'YDir','reverse');
xlim(ax,[-1 1]);
set(ax,'XTick',[]);

xlabel(ax,'Prior type');
ylabel(ax,'SNR (dB)');
zlabel(ax,'NMSE in $\mathbf{X}$');

view(ax,-35,18);
pbaspect(ax,[3 1.3 1.5]);

% 面板文字放在右下方
priorText = {'\textbf{full prior}','\textbf{partial prior}','\textbf{no prior}'};
yText = maxSNR - 0.02*(maxSNR - minSNR);
zText = zmin * 1.8;
for pt = 1:3
    text(ax, xPanel(pt), yText, zText, priorText{pt}, ...
        'FontWeight','bold', ...
        'Color',[0.25 0.25 0.25], ...
        'HorizontalAlignment','right', ...
        'VerticalAlignment','bottom');
end

legend(ax, hLeg, {'VAMP','QD-VAMP','PnP-VAMP-QD'}, ...
    'Location','northoutside', ...
    'Orientation','horizontal');
end

