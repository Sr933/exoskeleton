%% ================================================================
% Analysis: Publication-quality confusion matrix for CNN (mode = "all")
% - Loads results/cnn_predictions_all_seeds[_all].csv and summary
% - Uses test-only rows (isTrain==0)
% - Selects the best seed from cnn_accuracy_summary (or seed=1 fallback)
% - Plots row-normalized percentages with class names (if available)
% - Saves plots/confusion_matrix_cnn_all.(png|pdf)
% ================================================================
clear; clc;

% ---------------- Paths ----------------
scriptFullPath = mfilename('fullpath');
scriptDir = fileparts(scriptFullPath);
projectRoot = fileparts(scriptDir); % parent of "analysis"
resultsDir = fullfile(projectRoot, 'results');
plotsDir   = fullfile(projectRoot, 'plots');
if ~exist(plotsDir, 'dir'); mkdir(plotsDir); end

% ---------------- Locate files ----------------
predCandidates = {
    fullfile(resultsDir, 'cnn_predictions_all_seeds_all.csv')
    fullfile(resultsDir, 'cnn_all_predictions_all_seeds.csv')
    fullfile(resultsDir, 'cnn_predictions_all_seeds.csv')
};
predFile = '';
for i = 1:numel(predCandidates)
    f = predCandidates{i};
    if exist(f, 'file') == 2
        predFile = f; break;
    end
end
assert(~isempty(predFile), 'CNN predictions file not found in results/.');

sumCandidates = {
    fullfile(resultsDir, 'cnn_accuracy_summary_all.csv')
    fullfile(resultsDir, 'cnn_all_accuracy_summary.csv')
    fullfile(resultsDir, 'cnn_accuracy_summary.csv')
};
sumFile = '';
for i = 1:numel(sumCandidates)
    f = sumCandidates{i};
    if exist(f, 'file') == 2
        sumFile = f; break;
    end
end

fprintf('Using predictions: %s\n', predFile);
if ~isempty(sumFile), fprintf('Using summary:     %s\n', sumFile); end

% ---------------- Load data ----------------
Tp = local_read_table(predFile);
req = {'seed','isTrain','y_true','y_pred'};
missing = req(~ismember(req, Tp.Properties.VariableNames));
assert(isempty(missing), 'Missing columns in predictions: %s', strjoin(missing, ', '));

Tp.seed    = double(Tp.seed);
Tp.isTrain = double(Tp.isTrain);
Tp.y_true  = double(Tp.y_true);
Tp.y_pred  = double(Tp.y_pred);

Te = Tp(Tp.isTrain == 0, :);
assert(~isempty(Te), 'No test rows found in predictions file.');

% ---------------- Build confusion matrix aggregated over all seeds ----------------
K = max([Te.y_true; Te.y_pred]);
order = 1:K;
seeds = unique(Te.seed);
C = zeros(K, K);
for i = 1:numel(seeds)
    rows = Te(Te.seed == seeds(i), :);
    if isempty(rows), continue; end
    Ci = confusionmat(rows.y_true, rows.y_pred, 'Order', order);
    C = C + Ci;
end
fprintf('Aggregated %d seeds into confusion matrix.\n', numel(seeds));

% Row-normalize to percentages
rowSums = sum(C, 2);
Cperc = 100 * (C ./ max(rowSums, 1));

% Class names from library mapping (preferred)
classNames = get_class_names(order);
classNames = clean_axis_labels(classNames);  % replace underscores with spaces

fprintf('Class mapping:\n');
for i=1:numel(order)
    fprintf('  %d -> %s\n', order(i), classNames{i});
end
% Shortened labels for in-cell display on diagonal
diagLabels = shorten_labels(classNames, 12);

% ---------------- Plot (Nature quality) ----------------
% Single-column figure size (≈ 90 mm)
K = size(C,1);
figW = 3.54;                 % inches (~90 mm)
figH = max(3.0, min(6.0, 2.6 + 0.12*K));  % scale height with classes
f = figure('Color','w','Units','inches','Position',[1 1 figW figH]);

ax = axes(f);

% Palette: 'blues' (default), 'parula', or 'turbo'
palette = 'blues';
imagesc(ax, Cperc, [0 100]); axis(ax, 'image');
colormap(ax, make_seq_colormap(palette, 256));
caxis(ax, [0 100]);

cb = colorbar(ax);
cb.Location = 'eastoutside';
cb.Ticks = 0:20:100;
cb.Label.String = 'Row-normalized accuracy (%)';
cb.Label.FontName = 'Helvetica';
cb.Label.FontSize = 15;

% Ticks and labels
set(ax, 'XTick', 1:K, 'YTick', 1:K, ...
    'XTickLabel', shorten_labels(classNames, 22), ...
    'YTickLabel', shorten_labels(classNames, 28), ...
    'TickLabelInterpreter','none', ...
    'FontName','Arial', 'FontSize',8.5, 'LineWidth',1.0, ...
    'XAxisLocation','top', 'TickDir','out');
xtickangle(ax, 35);
xlabel(ax, 'Predicted class', 'FontName','Arial', 'FontSize',9);
ylabel(ax, 'True class',      'FontName','Arial', 'FontSize',9);

% Grid lines between cells (subtle)
hold(ax, 'on');
for k = 0.5:1:(K+0.5)
    plot(ax, [0.5 K+0.5], [k k], 'Color',[1 1 1]*0.82, 'LineWidth',0.5);
    plot(ax, [k k], [0.5 K+0.5], 'Color',[1 1 1]*0.82, 'LineWidth',0.5);
end
box(ax, 'off');

% Overlay percentage text with contrast-aware color
for i = 1:K
    for j = 1:K
        val = Cperc(i,j); if ~isfinite(val), val = 0; end
        valR = round(val, 2);
        txt = sprintf('%.2f', valR);  % percentage only in all cells
        tcol = [0 0 0]; if val >= 55, tcol = [1 1 1]; end
        text(j, i, txt, 'HorizontalAlignment','center', 'VerticalAlignment','middle', ...
            'FontSize', 6.5, 'FontWeight','bold', 'Color', tcol, 'FontName','Arial', ...
            'Interpreter','none');
    end
end

% Tight layout
axis(ax, [0.5 K+0.5 0.5 K+0.5]);
pad = 0.02;
outerpos = ax.OuterPosition; ti = ax.TightInset;
ax.Position = [outerpos(1)+ti(1)+pad, outerpos(2)+ti(2)+pad, ...
               1 - (ti(1)+ti(3)+pad*2), 1 - (ti(2)+ti(4)+pad*2)];

% ---------------- Save (vector + high-res raster) ----------------
pngPath = fullfile(plotsDir, 'confusion_matrix_cnn_all.png');
pdfPath = fullfile(plotsDir, 'confusion_matrix_cnn_all.pdf');
try
    set(f, 'InvertHardcopy','off', 'PaperPositionMode','auto');
    print(f, '-dpng', '-r600', pngPath);
    print(f, '-dpdf', '-painters', pdfPath);
    fprintf('Saved: %s\nSaved: %s\n', pngPath, pdfPath);
catch ME
    warning(ME.identifier, 'Failed to export confusion matrix: %s', ME.message);
end

%% ---------------- Helpers ----------------
function names = get_class_names_from_csv(T, order)
% Infer class names per numeric code using the CSV "folder" column.
% Picks the most frequent folder name for y_true==code; fallback to y_pred; else "Class <code>".
    K = numel(order);
    namesS = strings(1, K);
    hasFolder = ismember('folder', T.Properties.VariableNames);
    folders = strings(height(T),1);
    if hasFolder
        try
            folders = string(T.folder);
        catch
            folders(:) = "";
        end
    end
    for i = 1:K
        c = order(i);
        idx = T.y_true == c;
        if ~any(idx)
            idx = T.y_pred == c;
        end
        vals = strtrim(folders(idx));
        vals = vals(vals ~= "");
        if isempty(vals)
            namesS(i) = "Class " + string(c);
        else
            cats = categorical(vals);
            u = categories(cats);
            counts = countcats(cats);
            [~, j] = max(counts);
            namesS(i) = string(u{j});
        end
    end
    names = cellstr(namesS);
end

function T = local_read_table(file)
    try
        T = readtable(file, 'PreserveVariableNames', true, 'FileType','text');
        return;
    catch
    end
    fid = fopen(file, 'r');
    assert(fid~=-1, 'Unable to open %s', file);
    headerLine = fgetl(fid);
    assert(ischar(headerLine), 'File %s appears invalid.', file);
    headers = strsplit(headerLine, {',',';','\t'});
    fmt = repmat('%s', 1, numel(headers));
    C = textscan(fid, fmt, 'Delimiter', {',',';','\t'}, 'EndOfLine','\n');
    fclose(fid);
    T = table();
    for i=1:numel(headers)
        T.(matlab.lang.makeValidName(strtrim(headers{i}))) = string(C{i});
    end
    % Coerce known numeric columns
    for n = {'seed','isTrain','y_true','y_pred','accuracy'}
        if ismember(n{1}, T.Properties.VariableNames)
            T.(n{1}) = double(str2double(T.(n{1})));
        end
    end
end

function names = get_class_names(order)
% Prefer names from exoskeleton_library.label_mapping()
% Works when mapping is name->code (containers.Map) as in your project.
    names = arrayfun(@(k) sprintf('Class %d', k), order, 'UniformOutput', false);
    try
        lm = exoskeleton_library.label_mapping();
        if isa(lm, 'containers.Map')
            ks = lm.keys;
            vs = lm.values;
            for i = 1:numel(ks)
                clsName = string(ks{i});
                codeVal = double(vs{i});
                idx = find(order == codeVal, 1);
                if ~isempty(idx)
                    names{idx} = char(clsName);
                end
            end
        elseif isstruct(lm) && isfield(lm, 'code') && isfield(lm, 'name')
            [~, idx] = ismember(order, double(lm.code));
            nm = string({lm.name});
            good = idx > 0;
            names(good) = cellstr(nm(idx(good)));
        elseif isstruct(lm) && isfield(lm, 'classes')
            cls = lm.classes;
            if isstruct(cls) && all(isfield(cls, {'code','name'}))
                [~, idx] = ismember(order, double([cls.code]));
                nm = string({cls.name});
                good = idx > 0;
                names(good) = cellstr(nm(idx(good)));
            end
        end
    catch
        % keep defaults
    end
end

function out = shorten_labels(names, maxLen)
% Truncate long labels with ellipsis (non-destructive for tick layout)
    if nargin < 2, maxLen = 22; end
    out = names;
    for i = 1:numel(out)
        s = string(out{i});
        if strlength(s) > maxLen
            out{i} = char(extractBetween(s, 1, maxLen-1) + "…");
        end
    end
end

%% ---------------- Helpers (add below existing helpers) ----------------
function cm = make_seq_colormap(kind, n)
% Nature-style sequential colormaps without external deps.
    if nargin < 2, n = 256; end
    kind = lower(string(kind));
    switch kind
        case "blues"
            anchors = [  % ColorBrewer Blues (approx), light->dark
                0.9686 0.9843 1.0000  % #F7FBFF
                0.8706 0.9216 0.9686  % #DEEBF7
                0.7765 0.8588 0.9373  % #C6DBEF
                0.6196 0.7922 0.8824  % #9ECAE1
                0.4196 0.6824 0.8392  % #6BAED6
                0.1922 0.5098 0.7412  % #3182BD
                0.0314 0.3176 0.6118  % #08519C
            ];
            cm = interp1(linspace(0,1,size(anchors,1)), anchors, linspace(0,1,n), 'pchip');
        case "parula"
            cm = parula(n);
        case "turbo"
            if exist('turbo','file') == 2
                cm = turbo(n);
            else
                cm = parula(n);
            end
        otherwise
            cm = parula(n);
    end
end

function out = clean_axis_labels(names)
% Replace one or more underscores with a single space for axis display
    out = names;
    for i = 1:numel(out)
        s = string(out{i});
        s = regexprep(s, '_+', ' ');
        out{i} = char(strtrim(s));
    end
end