%% ================================================================
% Analysis: Publication-quality accuracy plot for Transfer Learning
% Variants: fine-tuned, pretrained-only, fine-tune-only
% Adds Random (lower bound) and CNN(all data) (upper bound) as references
% Saves high-res PNG and vector PDF to plots/.
% ================================================================
clear; clc;

% --------------- Resolve paths dynamically ----------------
scriptFullPath = mfilename('fullpath');
scriptDir = fileparts(scriptFullPath);
projectRoot = fileparts(scriptDir); % parent of "analysis"
resultsDir = fullfile(projectRoot, 'results');
plotsDir   = fullfile(projectRoot, 'plots');
if ~exist(plotsDir, 'dir'); mkdir(plotsDir); end

% --------------- Load TL summary ------------------------------
sumFile = fullfile(resultsDir, 'transfer_learning_summary.csv');
assert(exist(sumFile,'file')==2, 'Summary CSV not found: %s', sumFile);
T = local_read_table(sumFile);

% Coerce/rename expected columns if needed
req = {'subject','acc_fine_tuned','acc_pretrained_only','acc_ft_only','nFineTune'};
missing = req(~ismember(req, T.Properties.VariableNames));
assert(isempty(missing), 'Missing required columns in summary CSV: %s', strjoin(missing, ', '));

ft  = double(T.acc_fine_tuned);           % per subject
pre = double(T.acc_pretrained_only);
fto = double(T.acc_ft_only);

% Remove NaN subjects (if some subjects had no test samples)
mask = ~(isnan(ft) & isnan(pre) & isnan(fto));
ft  = ft(mask);
pre = pre(mask);
fto = fto(mask);

% Convert to percentage
ftP  = ft(:)  * 100;
preP = pre(:) * 100;
ftoP = fto(:) * 100;

% --------------- Load baselines using the a) accuracy plot data ----------
mode = "all";  % use the same mode as a_accuracy_cm
cnnAcc  = load_acc_for_model_mode(resultsDir, "cnn",    mode);   % [0..1]
randAcc = load_acc_for_model_mode(resultsDir, "random", mode);   % [0..1]
cnnAcc  = cnnAcc(:)  * 100;  % to %
randAcc = randAcc(:) * 100;  % to %

% Compute means and SEMs
% Replace the 3-series setup with a 5-series (add Random and CNN as bars)
seriesLabels = {'Random','Pretrained-only','Fine-tuned','FT-only','CNN (all data)'};

% All inputs in percentage
accSeries = { ...
    randAcc(:), ...   % Random (seeds) in %
    preP(:),   ...    % TL pretrained-only (subjects) in %
    ftP(:),    ...    % TL fine-tuned (subjects) in %
    ftoP(:),   ...    % TL fine-tune-only (subjects) in %
    cnnAcc(:)  ...    % CNN full (seeds) in %
};

means = nan(1, numel(accSeries));
sems  = nan(1, numel(accSeries));
stds  = nan(1, numel(accSeries));  % std for console reporting
for i=1:numel(accSeries)
    a = accSeries{i};
    a = a(isfinite(a)); % drop NaNs
    if ~isempty(a)
        means(i) = mean(a);
        stds(i)  = std(a);
        if numel(a) > 1
            sems(i) = std(a) / sqrt(numel(a));
        else
            sems(i) = 0;
            stds(i) = 0;
        end
    end
end

% --------------- Plot: Accuracy summary (Nature style) --------
f1 = figure('Color','w','Units','inches');
f1.Position = [1 1 7.2 3.8];
ax = axes(f1);
hold(ax, 'on');

% Colors for 5 bars
cols = [
    0.15 0.15 0.15;  % Random (dark grey)
    0.35 0.35 0.35;  % Pretrained-only (grey)
    0.20 0.44 0.72;  % Fine-tuned (blue accent)
    0.60 0.60 0.60;  % FT-only (light grey)
    0.75 0.75 0.75   % CNN (all data) (very light grey)
];

% Bars
bh = bar(ax, 1:numel(seriesLabels), means, 0.65, 'FaceColor','flat', 'EdgeColor','none');
for i=1:numel(seriesLabels)
    bh.CData(i,:) = cols(i,:);
end

% Error bars (SEM)
errorbar(ax, 1:numel(seriesLabels), means, sems, 'k', ...
    'LineStyle','none', 'LineWidth',1.5, 'CapSize',8);

% Overlay individual points per series (jitter)
for i=1:numel(accSeries)
    a = accSeries{i};
    a = a(isfinite(a));
    if isempty(a), continue; end
    xj = i + (rand(size(a))-0.5)*0.18;
    scatter(ax, xj, a, 24, cols(i,:), 'filled', ...
        'MarkerFaceAlpha',0.7, 'MarkerEdgeColor','k', 'LineWidth',0.3);
end

% Aesthetics
ax.FontName = 'Arial';
ax.FontSize = 13;
ax.LineWidth = 1.25;
ax.XTick = 1:numel(seriesLabels);
ax.XTickLabel = seriesLabels;
ylabel(ax, 'Accuracy (%)', 'FontName','Arial', 'FontSize',13);
ylim(ax, [0 100]);
yticks(ax, 0:20:100);
box(ax, 'off');

% Save figures (vector + high-res raster)
acc_png = fullfile(plotsDir, 'transfer_learning_accuracy.png');
acc_pdf = fullfile(plotsDir, 'transfer_learning_accuracy.pdf');
save_fig(f1, ax, acc_png, acc_pdf);

% -------- Console prints: mean ± std for each series --------
for i = 1:numel(accSeries)
    a = accSeries{i};
    a = a(isfinite(a));
    if isempty(a), continue; end
    fprintf('%-18s: %.2f ± %.2f %% (n=%d)\n', seriesLabels{i}, means(i), stds(i), numel(a));
end

%% ---------------- Local helpers ------------------------------
function A = load_acc_for_model_mode(resultsDir, model, mode)
% Same loader used by a_accuracy_cm: tries multiple naming patterns.
% Returns numeric accuracy vector in [0,1]. Empty if not found.
    model = string(model); mode = string(mode);
    candidates = strings(0);
    candidates(end+1) = fullfile(resultsDir, sprintf('%s_accuracy_summary_%s.csv', model, mode));
    candidates(end+1) = fullfile(resultsDir, sprintf('%s_%s_accuracy_summary.csv', model, mode));
    candidates(end+1) = fullfile(resultsDir, sprintf('%s_accuracy_summary.csv', model));
    for i = 1:numel(candidates)
        f = candidates(i);
        if exist(f, 'file')
            try
                T = readtable(f);
            catch
                T = local_read_table(f);
            end
            if ismember('accuracy', T.Properties.VariableNames)
                A = double(T.accuracy);
                return;
            end
        end
    end
    warning('Accuracy file for model %s (mode %s) not found.', model, mode);
    A = [];
end

function T = local_read_table(file)
    % Robust CSV reader: prefer readtable(FileType='text'), then manual parse
    try
        T = readtable(file, 'PreserveVariableNames', true, 'FileType','text');
        return;
    catch
        % fall through
    end
    % Manual CSV parse using textscan
    fid = fopen(file, 'r');
    if fid == -1
        error('Unable to open file %s', file);
    end
    headerLine = fgetl(fid);
    if ~ischar(headerLine)
        fclose(fid);
        error('File %s appears empty or invalid.', file);
    end
    headers = strsplit(headerLine, {',',';','\t'});
    numCols = numel(headers);
    fmt = repmat('%s', 1, numCols);
    C = textscan(fid, fmt, 'Delimiter', {',',';','\t'}, 'EndOfLine','\n');
    fclose(fid);
    % Build table as strings, then coerce numeric columns by name if present
    T = table();
    for i=1:numCols
        col = C{i};
        T.(matlab.lang.makeValidName(strtrim(headers{i}))) = string(col);
    end
    % Coerce numeric columns
    mustNum = {'acc_fine_tuned','acc_pretrained_only','acc_ft_only','nFineTune','accuracy'};
    for i=1:numel(mustNum)
        n = mustNum{i};
        if ismember(n, T.Properties.VariableNames)
            T.(n) = double(str2double(T.(n)));
        end
    end
end

function save_fig(figH, axH, pngPath, pdfPath)
    % Disable toolbar to avoid it appearing in exports
    try
        axtoolbar(axH, 'off');
    catch
    end
    try
        set(figH, 'InvertHardcopy','off');
        set(figH, 'PaperPositionMode','auto');
        % PNG raster
        print(figH, '-dpng', '-r600', pngPath);
        % PDF vector
        print(figH, '-dpdf', '-painters', pdfPath);
    catch ME
        warning(ME.identifier, 'Failed to export figure: %s', ME.message);
    end
end
