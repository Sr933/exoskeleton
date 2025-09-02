%% ================================================================
% Analysis: CNN accuracy under simulated sensor failures
% - Scans results/ for cnn_broken_*_accuracy_summary.csv
% - Optionally includes baseline CNN (cnn_accuracy_summary.csv)
% - Plots mean +/- SEM with per-seed jitter points
% - Saves PNG and PDF to plots/
% ================================================================
clear; clc;

% --------------- Resolve paths ----------------
scriptFullPath = mfilename('fullpath');
scriptDir = fileparts(scriptFullPath);
projectRoot = fileparts(scriptDir); % parent of "analysis"
resultsDir = fullfile(projectRoot, 'results');
plotsDir   = fullfile(projectRoot, 'plots');
if ~exist(plotsDir, 'dir'); mkdir(plotsDir); end

% --------------- Discover scenarios ----------------
D = dir(fullfile(resultsDir, 'cnn_broken_*_accuracy_summary*.csv'));
scenarios = strings(0);
files = strings(0);
for i = 1:numel(D)
    m = regexp(D(i).name, '^cnn_broken_(.+?)_accuracy_summary.*\.csv$', 'tokens', 'once');
    if ~isempty(m)
        scenarios(end+1) = string(m{1}); %#ok<AGROW>
        files(end+1) = fullfile(resultsDir, D(i).name); %#ok<AGROW>
    end
end

% Optional baseline CNN (no broken channels)
baselineFile = fullfile(resultsDir, 'cnn_accuracy_summary.csv');
hasBaseline = exist(baselineFile, 'file') == 2;

% Preferred display order
prefOrder = ["baseline","emg_all","emg_1_4","emg_5_8","imu_1","imu_2","imu_3","imu_1_3"];
if hasBaseline
    scenarios = ["baseline", scenarios];
    files     = [baselineFile, files];
end
% Deduplicate scenarios keeping first occurrence
[scenarios, ia] = unique(scenarios, 'stable');
files = files(ia);

% Reorder by preferred order (existing only)
[~, orderIdx] = ismember(prefOrder, scenarios);
orderIdx = orderIdx(orderIdx > 0);
scenarios = scenarios(orderIdx);
files     = files(orderIdx);

assert(~isempty(scenarios), 'No broken-channel summary files found in %s', resultsDir);

% --------------- Load accuracies and compute stats --------------
means = nan(1, numel(scenarios));
sems  = nan(1, numel(scenarios));
stds  = nan(1, numel(scenarios));  % added: std per scenario
accList = cell(1, numel(scenarios)); % per-scenario per-seed accuracies (%)

for i = 1:numel(scenarios)
    T = local_read_table(files(i));
    assert(ismember('accuracy', T.Properties.VariableNames), 'File missing accuracy column: %s', files(i));
    a = double(T.accuracy(:)) * 100; % to %
    a = a(isfinite(a));
    accList{i} = a;
    if ~isempty(a)
        means(i) = mean(a);
        sems(i)  = (numel(a) > 1) * std(a) / max(1, sqrt(numel(a)));
        stds(i)  = std(a);  % added
    else
        means(i) = NaN; sems(i) = NaN;
        stds(i)  = NaN;     % added
    end
end

% Pretty labels (blue/grey palette applied below)
labels = arrayfun(@(s) local_label_for_scenario(char(s)), scenarios, 'UniformOutput', false);

% Colors per scenario (baseline + EMG + IMU)
colMap = containers.Map( ...
    {'baseline','emg_all','emg_1_4','emg_5_8','imu_1','imu_2','imu_3','imu_1_3'}, ...
    {[0.3 0.3 0.3], [0.80 0.25 0.33], [0.95 0.55 0.20], [0.98 0.78 0.20], ...
     [0.20 0.44 0.72], [0.20 0.60 0.20], [0.55 0.35 0.64], [0.18 0.62 0.71]});
cols = zeros(numel(scenarios), 3);
for i = 1:numel(scenarios)
    key = char(lower(scenarios(i)));
    if isKey(colMap, key)
        cols(i,:) = colMap(key);
    else
        cols(i,:) = 0.6; % fallback grey
    end
end

% Blue/Grey palette (cycled), error bars stay black
palette = [
    0.85 0.85 0.85  % light grey
    0.60 0.60 0.60  % mid grey
    0.35 0.35 0.35  % dark grey
    0.20 0.44 0.72  % blue
    0.33 0.55 0.80  % medium blue
    0.12 0.31 0.54  % deep blue
];
cols = zeros(numel(scenarios), 3);
for i = 1:numel(scenarios)
    cols(i,:) = palette(mod(i-1, size(palette,1)) + 1, :);
end

% --------------- Plot ----------------------------
f = figure('Color','w','Units','inches');
f.Position = [1 1 9.5 4.2];  % larger figure
ax = axes(f); hold(ax, 'on');

bh = bar(ax, 1:numel(scenarios), means, 0.65, 'FaceColor','flat', 'EdgeColor','none');
for i = 1:numel(scenarios)
    bh.CData(i,:) = cols(i,:);
end

errorbar(ax, 1:numel(scenarios), means, sems, 'k', ...
    'LineStyle','none', 'LineWidth',2.0, 'CapSize',10);

% Jittered per-seed points
for i = 1:numel(scenarios)
    a = accList{i};
    if isempty(a), continue; end
    xj = i + (rand(size(a)) - 0.5) * 0.18;
    scatter(ax, xj, a, 36, cols(i,:), 'filled', ...
        'MarkerFaceAlpha', 0.75, 'MarkerEdgeColor', 'k', 'LineWidth', 0.5);
end

% Style
ax.FontName = 'Helvetica';
ax.FontSize = 16;
ax.LineWidth = 1.6;
ax.XTick = 1:numel(scenarios);
ax.XTickLabel = labels;
ax.XTickLabelRotation = 15;
ylabel(ax, 'Accuracy (%)', 'FontName','Helvetica', 'FontSize',16);
ylim(ax, [0 100]);
yticks(ax, 0:20:100);
box(ax, 'off');

% Save
pngPath = fullfile(plotsDir, 'cnn_broken_channels_accuracy.png');
pdfPath = fullfile(plotsDir, 'cnn_broken_channels_accuracy.pdf');
save_fig(f, ax, pngPath, pdfPath);

% Console prints: mean ± std per scenario
for i = 1:numel(scenarios)
    a = accList{i};
    if isempty(a), continue; end
    fprintf('%-22s: %.2f ± %.2f %% (n=%d)\n', labels{i}, means(i), stds(i), numel(a));
end

% Also export a CSV summary with means/SEMs
outCsv = fullfile(resultsDir, 'cnn_broken_channels_accuracy_summary.csv');
Tsum = table(string(scenarios(:)), means(:), stds(:), sems(:), ...
    'VariableNames', {'scenario','mean_acc_pct','std_pct','sem_pct'});
try, writetable(Tsum, outCsv); catch, end

%% ---------------- Helpers ------------------------------
function T = local_read_table(file)
    try
        T = readtable(file, 'PreserveVariableNames', true, 'FileType','text');
        return;
    catch
    end
    fid = fopen(file, 'r');
    if fid == -1, error('Unable to open %s', file); end
    headerLine = fgetl(fid);
    if ~ischar(headerLine), fclose(fid); error('File %s appears invalid.', file); end
    headers = strsplit(headerLine, {',',';','\t'});
    fmt = repmat('%s', 1, numel(headers));
    C = textscan(fid, fmt, 'Delimiter', {',',';','\t'}, 'EndOfLine','\n');
    fclose(fid);
    T = table();
    for i=1:numel(headers)
        T.(matlab.lang.makeValidName(strtrim(headers{i}))) = string(C{i});
    end
    if ismember('accuracy', T.Properties.VariableNames)
        T.accuracy = double(str2double(T.accuracy));
    end
end

function lbl = local_label_for_scenario(s)
% Map scenario keys to clean labels using leg/channel definitions:
%   right_leg = IMU 1 & 3 + EMG 1–4
%   left_leg  = IMU 2     + EMG 5–8
    switch lower(string(s))
        case "baseline",   lbl = "Baseline";
        case "emg_all",    lbl = "All EMG";
        case "emg_1_4",    lbl = "Right foot EMG";
        case "emg_5_8",    lbl = "Left foot EMG";
        case "imu_1",      lbl = "Right shank IMU";
        case "imu_2",      lbl = "Left shank IMU";
        case "imu_3",      lbl = "Right foot IMU";
        case "imu_1_3",    lbl = "Right leg IMUs";
        otherwise
            % Fallback: replace underscores, title case
            s = strrep(char(s), '_', ' ');
             lbl = regexprep(s, '(^| )([a-z])', '${upper($2)}');
    end
end
%
function save_fig(figH, axH, pngPath, pdfPath)
    try
        axtoolbar(axH, 'off');
    catch
    end
    try
        set(figH, 'InvertHardcopy','off');
        set(figH, 'PaperPositionMode','auto');
        print(figH, '-dpng', '-r600', pngPath);
        print(figH, '-dpdf', '-painters', pdfPath);
    catch ME
        warning(ME.identifier, 'Failed to export figure: %s', ME.message);
    end
end