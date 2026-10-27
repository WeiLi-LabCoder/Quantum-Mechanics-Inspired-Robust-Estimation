clear; clc; close all;

%% =========================================================
% Horizontal robustness summary plot from .fig
% x-axis: NMSE (log)
% y-axis: methods
%% =========================================================

figFile = 'com_NMSE_X_HuberLAD_all.fig';

if ~isfile(figFile)
    error('File "%s" not found.', figFile);
end

%% 1) Read .fig
fig = openfig(figFile, 'invisible');
ax  = findobj(fig, 'Type', 'axes');
lines = flipud(findobj(ax, 'Type', 'line'));

if isempty(lines)
    close(fig);
    error('No line objects found in the .fig file.');
end

data = struct('name', {}, 'x', {}, 'y', {});
for k = 1:numel(lines)
    data(k).name = string(get(lines(k), 'DisplayName'));
    data(k).x    = get(lines(k), 'XData');
    data(k).y    = get(lines(k), 'YData');
end
close(fig);

fprintf('Extracted %d curves:\n', numel(data));
for k = 1:numel(data)
    fprintf('%2d: %s\n', k, data(k).name);
end

%% 2) Settings
center_snr = 15;

methods_plot = {'Huber', 'LAD', 'VAMP', 'QD-VAMP', 'PnP-VAMP-QD'};
methods_tick = {'Huber', 'LAD', 'VAMP', 'QD-VAMP', 'PnP-VAMP-QD'};

% Case order:
% 1) alpha=1, Nt=8  -> darkest
% 2) alpha=1, Nt=1  -> medium
% 3) alpha=2, Nt=1  -> lightest
scenarios = {
    struct('label','\alpha = 1, N_t = 8', 'keywords',{{'n_t=8', '\alpha=1'}}), ...
    struct('label','\alpha = 1, N_t = 1', 'keywords',{{'n_t=1', '\alpha=1'}}), ...
    struct('label','\alpha = 2, N_t = 1', 'keywords',{{'n_t=1', '\alpha=2'}})
};

nMethods = numel(methods_plot);
nScen    = numel(scenarios);

baseY    = 1:nMethods;
offsets  = [-0.22, 0, 0.22];

% geometry
maxHalfWidth   = 0.085;   % thickness at the two ends
waistHalfWidth = 0.020;   % thickness at the center

% line styles
edgeColor      = [0.30 0.30 0.30];
centerLineCol  = [0.18 0.18 0.18];

outlineLW      = 0.55;   % thin outer boundary
centerLW       = 0.90;   % center horizontal guide
waistLW        = 1.45;   % emphasized center waist line
capLW          = 2.10;   % emphasized end caps
capH           = 0.060;  % cap height in y-direction

%% 3) Corrected palette mapping
% Each row = one method
% Columns = [dark, medium, light]

methodCaseColors = zeros(nMethods, nScen, 3);

% Huber -> purple family
methodCaseColors(1,1,:) = hex2rgb('#734F87');
methodCaseColors(1,2,:) = hex2rgb('#8484B5');
methodCaseColors(1,3,:) = hex2rgb('#AAA9D6');

% LAD -> green family
methodCaseColors(2,1,:) = hex2rgb('#5F6929');
methodCaseColors(2,2,:) = hex2rgb('#B6CC7B');
methodCaseColors(2,3,:) = hex2rgb('#BDC6BE');

% VAMP -> orange family
methodCaseColors(3,1,:) = hex2rgb('#D17D11');
methodCaseColors(3,2,:) = hex2rgb('#DBA449');
methodCaseColors(3,3,:) = hex2rgb('#C9B584');

% QD-VAMP -> rose family
methodCaseColors(4,1,:) = hex2rgb('#B56253');
methodCaseColors(4,2,:) = hex2rgb('#D59A94');
methodCaseColors(4,3,:) = hex2rgb('#EAC1C6');

% PnP-VAMP -> blue family
methodCaseColors(5,1,:) = hex2rgb('#042159');
methodCaseColors(5,2,:) = hex2rgb('#0E4C98');
methodCaseColors(5,3,:) = hex2rgb('#91AED4');

%% 4) Helper
find_curve = @(keywords) find(arrayfun(@(d) ...
    all(cellfun(@(kw) contains(lower(char(d.name)), lower(kw)), keywords)), data));

%% 5) Collect stats
Y_center = nan(nMethods, nScen);
Y_min    = nan(nMethods, nScen);
Y_max    = nan(nMethods, nScen);
SNR_min  = nan(nMethods, nScen);
SNR_max  = nan(nMethods, nScen);

for i = 1:nMethods
    for j = 1:nScen
        kw = [{methods_plot{i}}, scenarios{j}.keywords];
        idx = find_curve(kw);

        if isempty(idx)
            warning('No match found for %s | %s', methods_plot{i}, scenarios{j}.label);
            continue;
        end

        d = data(idx(1));
        st = local_stats_full(d.x, d.y, center_snr);

        Y_center(i,j) = st.centerVal;
        Y_min(i,j)    = st.minVal;
        Y_max(i,j)    = st.maxVal;
        SNR_min(i,j)  = st.minSNR;
        SNR_max(i,j)  = st.maxSNR;
    end
end

%% 6) Figure
f = figure('Color', 'w', 'Position', [100 100 1100 520]);
axNew = axes(f);
hold(axNew, 'on');
box(axNew, 'off');
grid(axNew, 'on');

set(axNew, 'XScale', 'log');
set(axNew, 'Color', 'w');
set(axNew, 'Layer', 'top');
set(axNew, 'FontName', 'Times New Roman');
set(axNew, 'FontSize', 11);

% cleaner grid
axNew.GridColor = [0.88 0.88 0.88];
axNew.GridAlpha = 0.30;
axNew.XMinorGrid = 'on';
axNew.YMinorGrid = 'off';
axNew.YGrid = 'off';

xl = [1e-6, 1e6];
xlim(xl);
ylim([0.5, nMethods + 0.5]);

%% 7) Draw horizontal hourglasses
for i = 1:nMethods
    for j = 1:nScen
        if any(isnan([Y_min(i,j), Y_center(i,j), Y_max(i,j)]))
            continue;
        end

        y0   = baseY(i) + offsets(j);   % method position on y-axis
        c    = squeeze(methodCaseColors(i,j,:))';

        xmin = Y_min(i,j);
        xc   = Y_center(i,j);
        xmax = Y_max(i,j);

        xc = min(max(xc, xmin), xmax);

        % build horizontal hourglass polygon
        [Xpoly, Ypoly] = build_hourglass_polygon_horizontal( ...
            y0, xmin, xc, xmax, maxHalfWidth, waistHalfWidth);

        % body
        patch('XData', Xpoly, 'YData', Ypoly, ...
              'FaceColor', c, ...
              'EdgeColor', edgeColor, ...
              'LineWidth', outlineLW, ...
              'FaceAlpha', 0.88);

        % center horizontal guide
        plot([xmin xmax], [y0 y0], '-', ...
            'Color', centerLineCol, 'LineWidth', centerLW);

        % waist marker
        plot([xc xc], [y0-waistHalfWidth*0.90, y0+waistHalfWidth*0.90], ...
            '-', 'Color', centerLineCol, 'LineWidth', waistLW);

        % small waist dot
        plot(xc, y0, 'o', ...
            'MarkerSize', 2.8, ...
            'MarkerFaceColor', [1 1 1], ...
            'MarkerEdgeColor', centerLineCol, ...
            'LineWidth', 0.7);

        % end caps
        plot([xmax xmax], [y0-capH, y0+capH], '-', ...
            'Color', centerLineCol, 'LineWidth', capLW);
        plot([xmin xmin], [y0-capH, y0+capH], '-', ...
            'Color', centerLineCol, 'LineWidth', capLW);
    end
end

%% 8) Axes / labels
yticks(baseY);
yticklabels([]);

 

xlabel('NMSE', 'FontName', 'Times New Roman', 'FontSize', 14);
ylabel('');

% Put first method at the top
set(gca, 'YDir', 'reverse');

%% 9) Legend with corrected dark-to-light mapping
legendSwatches = [
    0.35 0.35 0.35;   % darkest -> alpha=1, Nt=8
    0.60 0.60 0.60;   % medium  -> alpha=1, Nt=1
    0.85 0.85 0.85    % lightest-> alpha=2, Nt=1
];

hleg = gobjects(1, nScen);
lineStyles = {'-', '--', ':'};

for j = 1:nScen
    hleg(j) = patch(nan, nan, legendSwatches(j,:), ...
        'EdgeColor', edgeColor, ...
        'LineWidth', 1.2, ...
        'LineStyle', lineStyles{j}, ...   % ⭐ 关键
        'FaceAlpha', 1.0);
end

lgd = legend(hleg, ...
    {'$\alpha = 1, N_t = 8$', '$\alpha = 1, N_t = 1$', '$\alpha = 2, N_t = 1$'}, ...
    'Location', 'northoutside', ...
    'Orientation', 'horizontal', ...
    'NumColumns', 1, ...
    'Interpreter', 'tex', ...
    'FontSize', 11, ...
    'Box', 'on');
lgd.EdgeColor = [0.78 0.78 0.78];
lgd.Color = [1 1 1];

%% 10) Annotation
annotation('textbox', [0.15 0.80 0.28 0.11], ...
    'String', { ...
        'Right: maximum NMSE over SNR', ...
        'Waist: NMSE at 15 dB', ...
        'Left: minimum NMSE over SNR', ...
        'SNR range: [-5, 20] dB'}, ...
    'FitBoxToText', 'on', ...
    'BackgroundColor', [1 1 1], ...
    'EdgeColor', [0.85 0.85 0.85], ...
    'FontName', 'Times New Roman', ...
    'FontSize', 10);

%% 11) Export
set(f, 'Renderer', 'painters');
set(gca, 'LooseInset', get(gca,'TightInset'));

exportgraphics(gcf, 'fig_NMSE_X_HuberLAD.pdf', ...
    'ContentType', 'vector', ...
    'BackgroundColor', 'white');
 

fprintf('\nSaved:\n');
fprintf('  fig_NMSE_X_HuberLAD_horizontal.pdf\n');
fprintf('  fig_NMSE_X_HuberLAD_horizontal.png\n');

%% =========================================================
% Local functions
%% =========================================================

function st = local_stats_full(x, y, center_snr)
    x = x(:);
    y = y(:);

    valid = isfinite(x) & isfinite(y) & (y > 0);
    x = x(valid);
    y = y(valid);

    if isempty(x)
        st.minVal = NaN;
        st.maxVal = NaN;
        st.centerVal = NaN;
        st.minSNR = NaN;
        st.maxSNR = NaN;
        return;
    end

    [st.minVal, idxMin] = min(y);
    [st.maxVal, idxMax] = max(y);
    st.minSNR = x(idxMin);
    st.maxSNR = x(idxMax);

    [~, idxCenter] = min(abs(x - center_snr));
    st.centerVal = y(idxCenter);
end

function rgb = hex2rgb(hexStr)
    hexStr = char(hexStr);
    if hexStr(1) == '#'
        hexStr = hexStr(2:end);
    end
    if numel(hexStr) ~= 6
        error('hex color must have 6 characters.');
    end
    rgb = [hex2dec(hexStr(1:2)), hex2dec(hexStr(3:4)), hex2dec(hexStr(5:6))] / 255;
end

function [Xpoly, Ypoly] = build_hourglass_polygon_horizontal(y0, xmin, xc, xmax, hmax, hmin)
    % Horizontal hourglass:
    % x-axis: value
    % y-axis: thickness around y0
    %
    % left end  -> xmin
    % waist     -> xc
    % right end -> xmax

    n1 = 80;
    n2 = 80;

    lx_left  = linspace(log10(xmin), log10(xc), n1);
    lx_right = linspace(log10(xc),   log10(xmax), n2);

    tl = linspace(1, 0, n1);   % left end -> waist
    tr = linspace(0, 1, n2);   % waist -> right end

    p = 0.72;
    hl = hmin + (hmax - hmin) * (tl .^ p);
    hr = hmin + (hmax - hmin) * (tr .^ p);

    xl = 10.^lx_left;
    xr = 10.^lx_right;

    % upper boundary: left -> waist -> right
    xUpper = [xl, xr(2:end)];
    yUpper = [y0 + hl, y0 + hr(2:end)];

    % lower boundary: right -> waist -> left
    xLower = [fliplr(xr), fliplr(xl(1:end-1))];
    yLower = [y0 - fliplr(hr), y0 - fliplr(hl(1:end-1))];

    Xpoly = [xUpper, xLower];
    Ypoly = [yUpper, yLower];
end