%% ================================================================
% Analysis: Time-series CNN accuracy (percentage of time) per channel mode
% - Loads:
%     results/timeseries_cnn_time_accuracy_<mode>_shift*.csv   (new)
%     results/timeseries_cnn_accuracy_<mode>_stride*.csv       (alt)
%   Back-compat for "all":
%     results/timeseries_cnn_time_accuracy_shift*.csv          (legacy)
% - For each mode: plots grouped bars (Raw vs Kalman) per stride with SEM+jitter
% - Saves PNG and PDF to plots/, and a summary CSV per mode
% ================================================================
clear; clc;

% --------------- Paths ----------------
scriptFullPath = mfilename('fullpath');
scriptDir = fileparts(scriptFullPath);
projectRoot = fileparts(scriptDir);
resultsDir = fullfile(projectRoot, 'results');
plotsDir   = fullfile(projectRoot, 'plots');
if ~exist(plotsDir, 'dir'); mkdir(plotsDir); end

channelModes = ["all","left_leg","right_leg"];
foundAny = false;

% Collect per-mode data for a combined 6-bar plot
modesCollected = strings(0,1);
stPerMode = {};          % cell of numeric stride arrays per mode
rawPerMode = {};         % cell of cell-arrays: per-stride vectors
kalPerMode = {};         % cell of cell-arrays: per-stride vectors

for mode = channelModes
    % --------------- Discover stride files for this mode ---------------
    [st, files] = discover_mode_files(resultsDir, mode);

    % Back-compat: legacy filenames without mode for "all"
    if isempty(st) && mode == "all"
        D = dir(fullfile(resultsDir, 'timeseries_cnn_time_accuracy_shift*.csv'));
        strides = []; files0 = strings(0,1);
        for i = 1:numel(D)
            m = regexp(D(i).name, 'shift(\d+)\.csv$', 'tokens','once');
            if ~isempty(m)
                strides(end+1,1) = str2double(m{1}); %#ok<AGROW>
                files0(end+1,1)   = fullfile(resultsDir, D(i).name); %#ok<AGROW>
            end
        end
        [st, order] = sort(strides); files = files0(order);
    end

    if isempty(st)
        warning('No time-series accuracy files found for mode "%s" in %s', mode, resultsDir);
        continue;
    end
    foundAny = true;

    % --------------- Load and aggregate ----------------
    nG = numel(st);
    means_raw = nan(nG,1); sems_raw = nan(nG,1);
    means_kal = nan(nG,1); sems_kal = nan(nG,1);
    rawList = cell(nG,1); kalList = cell(nG,1);

    for i = 1:nG
        T = local_read_table(files(i));
        need = {'accuracy_raw_pct','accuracy_kalman_pct'};
        missing = need(~ismember(need, T.Properties.VariableNames));
        assert(isempty(missing), 'File %s missing columns: %s', files(i), strjoin(missing, ', '));

        aRaw = double(T.accuracy_raw_pct(:));
        aKal = double(T.accuracy_kalman_pct(:));
        aRaw = aRaw(isfinite(aRaw));
        aKal = aKal(isfinite(aKal));
        rawList{i} = aRaw; kalList{i} = aKal;

        [means_raw(i), sems_raw(i)] = msem(aRaw);
        [means_kal(i), sems_kal(i)] = msem(aKal);
    end

    % Skip per-mode plots and files; only collect data for the combined plot.
    % --------------- Save ----------------
    % Per-mode summary CSV (means±SEM)
    sumCsv = fullfile(resultsDir, sprintf('timeseries_cnn_accuracy_summary_%s.csv', mode));
    Tsum = table(st(:), means_raw(:), sems_raw(:), means_kal(:), sems_kal(:), ...
        'VariableNames', {'stride','mean_raw_pct','sem_raw_pct','mean_kalman_pct','sem_kalman_pct'});
    try, writetable(Tsum, sumCsv); catch, end
    fprintf('Saved: %s\n', sumCsv);

    % Store for combined plot
    modesCollected(end+1,1) = mode;                  %#ok<AGROW>
    stPerMode{end+1,1}  = st;                        %#ok<AGROW>
    rawPerMode{end+1,1} = rawList;                   %#ok<AGROW>
    kalPerMode{end+1,1} = kalList;                   %#ok<AGROW>
end

% Combined 6-bar plot: Raw vs Kalman for all three modes
if foundAny && ~isempty(modesCollected)
    % Prefer a common stride across all modes (1000, then 2000)
    prefs = [1000 2000];
    commonStride = NaN;
    for p = prefs
        if all(cellfun(@(v) any(v==p), stPerMode))
            commonStride = p; break;
        end
    end

    nM = numel(modesCollected);
    means_raw_m = nan(nM,1); sems_raw_m = nan(nM,1);
    means_kal_m = nan(nM,1); sems_kal_m = nan(nM,1);
    xlabels = strings(nM,1);
    stride_used = nan(nM,1);
    idxPerMode = zeros(nM,1);

    for i = 1:nM
        st = stPerMode{i};
        rl = rawPerMode{i};
        kl = kalPerMode{i};

        % Map mode to display name
        dispName = mode_display_name(modesCollected(i));

        if ~isnan(commonStride)
            idx = find(st==commonStride, 1);
            stride_used(i) = commonStride;
            xlabels(i) = dispName;
        else
            % fallback: pick smallest stride available per mode and annotate
            [~, idx] = min(st);
            stride_used(i) = st(idx);
            xlabels(i) = sprintf('%s (stride %d)', dispName, st(idx));
        end
        idxPerMode(i) = idx;

        aRaw = rl{idx};
        aKal = kl{idx};
        [means_raw_m(i), sems_raw_m(i)] = msem(aRaw);
        [means_kal_m(i), sems_kal_m(i)] = msem(aKal);
    end

    % Plot 6 bars: 3 groups (modes) x 2 bars (Raw, Kalman)
    fC = figure('Color','w','Units','inches'); 
    fC.Position = [1 1 8.5 4.4];
    axC = axes(fC); hold(axC,'on');

    YC = [means_raw_m, means_kal_m];
    bhC = bar(axC, YC, 'grouped', 'EdgeColor','none', 'BarWidth',0.65);

    cols = [
        0.55 0.55 0.55;  % Raw (grey)
        0.20 0.44 0.72;  % Kalman (blue)
    ];
    for j = 1:numel(bhC)
        bhC(j).FaceColor = cols(j,:);
    end

    useXEP = all(arrayfun(@(h) isprop(h,'XEndPoints'), bhC));
    if useXEP
        xRaw = bhC(1).XEndPoints; xKal = bhC(2).XEndPoints;
    else
        ng = size(YC,1); ns = size(YC,2);
        groupWidth = min(0.8, ns/(ns+1.5));
        x0 = 1:ng; offset = linspace(-groupWidth/2, groupWidth/2, ns);
        xRaw = x0 + offset(1); xKal = x0 + offset(2);
    end

    errorbar(axC, xRaw, means_raw_m, sems_raw_m, 'k', 'LineStyle','none', 'LineWidth',1.8, 'CapSize',10);
    errorbar(axC, xKal, means_kal_m, sems_kal_m, 'k', 'LineStyle','none', 'LineWidth',1.8, 'CapSize',10);

    % Jittered per-seed points on top of each bar
    jitter = 0.12;
    for i = 1:nM
        aR = rawPerMode{i}{idxPerMode(i)};
        if ~isempty(aR)
            xj = xRaw(i) + (rand(size(aR))-0.5)*jitter;
            scatter(axC, xj, aR, 24, [0.55 0.55 0.55], 'filled', ...
                'MarkerFaceAlpha',0.7, 'MarkerEdgeColor','k', 'LineWidth',0.3);
        end
        aK = kalPerMode{i}{idxPerMode(i)};
        if ~isempty(aK)
            xj = xKal(i) + (rand(size(aK))-0.5)*jitter;
            scatter(axC, xj, aK, 24, [0.20 0.44 0.72], 'filled', ...
                'MarkerFaceAlpha',0.7, 'MarkerEdgeColor','k', 'LineWidth',0.3);
        end
    end

    % Style
    axC.FontName = 'Helvetica';
    axC.FontSize = 16;
    axC.LineWidth = 1.6;
    axC.XTick = 1:nM;
    axC.XTickLabel = xlabels;
    ylabel(axC, 'Accuracy (% time correct)', 'FontName','Helvetica', 'FontSize',16);
    ylim(axC, [0 100]); yticks(axC, 0:20:100);
    lgC = legend(axC, {'Raw','Kalman-smoothed'}, 'Location','southoutside', 'Orientation','horizontal', 'Box','off');
    lgC.FontName = 'Helvetica'; lgC.FontSize = 14;

    box(axC, 'off');

    % Save combined figure and summary CSV
    pngAll = fullfile(plotsDir, 'timeseries_cnn_accuracy_all_modes.png');
    pdfAll = fullfile(plotsDir, 'timeseries_cnn_accuracy_all_modes.pdf');
    save_fig(fC, axC, pngAll, pdfAll);
    fprintf('Saved: %s\nSaved: %s\n', pngAll, pdfAll);

    % Also include human-friendly condition names in the CSV
    condNames = arrayfun(@(m) mode_display_name(m), modesCollected, 'UniformOutput', false);
    sumAllCsv = fullfile(resultsDir, 'timeseries_cnn_accuracy_summary_all_modes.csv');
    Tall = table(string(modesCollected), string(condNames(:)), stride_used, ...
                 means_raw_m, sems_raw_m, means_kal_m, sems_kal_m, ...
        'VariableNames', {'mode','condition','stride_used','mean_raw_pct','sem_raw_pct','mean_kalman_pct','sem_kalman_pct'});
    try, writetable(Tall, sumAllCsv); catch, end
    fprintf('Saved: %s\n', sumAllCsv);
end

if ~foundAny
    error('No time-series accuracy files found. Run i_timeseries_classification.m first.');
end

%% ---------------- Helpers ----------------
function [st, files] = discover_mode_files(resultsDir, mode)
% Find per-mode files saved by the generator scripts (support two patterns).
    st = []; files = strings(0,1);

    % Pattern A: timeseries_cnn_time_accuracy_<mode>_shiftXXXX.csv
    D1 = dir(fullfile(resultsDir, sprintf('timeseries_cnn_time_accuracy_%s_shift*.csv', mode)));
    for i = 1:numel(D1)
        m = regexp(D1(i).name, sprintf('^timeseries_cnn_time_accuracy_%s_shift(\\d+)\\.csv$', mode), 'tokens','once');
        if ~isempty(m)
            st(end+1,1) = str2double(m{1}); %#ok<AGROW>
            files(end+1,1) = fullfile(resultsDir, D1(i).name); %#ok<AGROW>
        end
    end

    % Pattern B: timeseries_cnn_accuracy_<mode>_strideXXXX.csv
    D2 = dir(fullfile(resultsDir, sprintf('timeseries_cnn_accuracy_%s_stride*.csv', mode)));
    for i = 1:numel(D2)
        m = regexp(D2(i).name, sprintf('^timeseries_cnn_accuracy_%s_stride(\\d+)\\.csv$', mode), 'tokens','once');
        if ~isempty(m)
            st(end+1,1) = str2double(m{1}); %#ok<AGROW>
            files(end+1,1) = fullfile(resultsDir, D2(i).name); %#ok<AGROW>
        end
    end

    % Deduplicate and sort
    if ~isempty(st)
        [st, idx] = sort(st);
        files = files(idx);
        [st, uidx] = unique(st, 'stable');
        files = files(uidx);
    end
end

function [m, s] = msem(x)
    x = x(isfinite(x));
    if isempty(x), m = NaN; s = NaN; return; end
    m = mean(x); n = numel(x);
    if n > 1, s = std(x) / sqrt(n); else, s = 0; end
end

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
        T.(matlab.lang.makeValidName(strtrim(headers{i}))) = string(C{i}); %#ok<AGROW>
    end
    % numeric coercion
    for n = {'seed','accuracy_raw_pct','accuracy_kalman_pct','n_windows'}
        if ismember(n{1}, T.Properties.VariableNames)
            T.(n{1}) = double(str2double(T.(n{1})));
        end
    end
end

function save_fig(figH, axH, pngPath, pdfPath)
    try, axtoolbar(axH, 'off'); catch, end
    try
        set(figH, 'InvertHardcopy','off', 'PaperPositionMode','auto');
        print(figH, '-dpng', '-r600', pngPath);
        print(figH, '-dpdf', '-painters', pdfPath);
    catch ME
        warning(ME.identifier, 'Failed to export figure: %s', ME.message);
    end
end

% Add this helper at end of file
function name = mode_display_name(mode)
% Map internal mode keys to publication-friendly names.
    mode = string(mode);
    switch mode
        case "all",        name = "Both Legs";
        case "left_leg",   name = "Left Leg";
        case "right_leg",  name = "Right Leg";
        otherwise,         name = char(mode);
    end
end