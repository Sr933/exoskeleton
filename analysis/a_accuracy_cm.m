%% ================================================================
% Analysis: Publication-quality accuracy plots per channel mode
% Modes: all, imu, emg. Compares Random, SVM, CNN, LSTM in each plot.
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

% --------------- Settings --------------------------
modes = ["all","imu","emg"];          % channel modes
models = ["random","svm","cnn","lstm"];
labels = {'Random','SVM','CNN','LSTM'};
% Blue/Grey palette (error bars remain black)
cols = [
    0.85 0.85 0.85;  % Random - light grey
    0.40 0.40 0.40;  % SVM - dark grey
    0.20 0.44 0.72;  % CNN - blue
    0.12 0.31 0.54   % LSTM - deep blue
];

% --------------- Plot: Accuracy per mode ----------------
for mi = 1:numel(modes)
    mode = modes(mi);

    % Load accuracy arrays for each model (flexible filenames)
    accs = cell(1, numel(models));
    for j = 1:numel(models)
        accs{j} = load_acc_for_model_mode(resultsDir, models(j), mode);
    end

    % Compute mean/SEM/STD in percentage
    means = nan(1, numel(accs));
    sems  = nan(1, numel(accs));
    stds  = nan(1, numel(accs));  % added
    for i = 1:numel(accs)
        a = accs{i}; a = a(:);
        if ~isempty(a)
            means(i) = mean(a) * 100;
            if numel(a) > 1
                stds(i) = std(a) * 100;          % added
                sems(i) = (stds(i) / sqrt(numel(a))); % updated to use stds
            else
                stds(i) = 0;                     % added
                sems(i) = 0;
            end
        end
    end

    % Figure
    f = figure('Color','w','Units','inches');
    f.Position = [1 1 7.8 4.3];  % larger figure
    ax = axes(f); hold(ax, 'on');

    bh = bar(ax, 1:numel(labels), means, 0.65, 'FaceColor','flat', 'EdgeColor','none');
    for i=1:numel(labels)
        bh.CData(i,:) = cols(i,:);
    end

    errorbar(ax, 1:numel(labels), means, sems, 'k', ...
        'LineStyle','none', 'LineWidth',2.0, 'CapSize',10);

    % Overlay per-seed points for each model (if present)
    for i=1:numel(accs)
        a = accs{i}; a = a(:) * 100;
        if isempty(a), continue; end
        xj = i + (rand(size(a))-0.5)*0.18;
        scatter(ax, xj, a, 36, cols(i,:), 'filled', ...
            'MarkerFaceAlpha',0.75, 'MarkerEdgeColor','k', 'LineWidth',0.5);
    end

    ax.FontName = 'Helvetica';
    ax.FontSize = 20;
    ax.LineWidth = 1.6;
    ax.XTick = 1:numel(labels);
    ax.XTickLabel = labels;
    ylabel(ax, 'Accuracy (%)', 'FontName','Helvetica', 'FontSize',20);
    ax.XTickLabelRotation = 0;
    ylim(ax, [0 100]);
    yticks(ax, 0:20:100);
    box(ax, 'off');

    % Save per-mode figures
    acc_png = fullfile(plotsDir, sprintf('accuracy_summary_%s.png', mode));
    acc_pdf = fullfile(plotsDir, sprintf('accuracy_summary_%s.pdf', mode));
    save_fig(f, ax, acc_png, acc_pdf);

    % -------- Console prints: CNN per mode, and all models for "all" --------
    idxCNN = find(models == "cnn", 1);
    if ~isempty(idxCNN) && ~isempty(accs{idxCNN})
        a = accs{idxCNN}; a = a(:) * 100; a = a(isfinite(a));
        if ~isempty(a)
            fprintf('[%s] CNN accuracy: %.2f ± %.2f %% (n=%d)\n', string(mode), mean(a), std(a), numel(a));
        end
    end

    if mode == "all"
        for j = 1:numel(models)
            aj = accs{j}; aj = aj(:) * 100; aj = aj(isfinite(aj));
            if isempty(aj), continue; end
            fprintf('[all] %s accuracy: %.2f ± %.2f %% (n=%d)\n', labels{j}, mean(aj), std(aj), numel(aj));
        end
    end
end

% --------------- Helpers ------------------------------
function A = load_acc_for_model_mode(resultsDir, model, mode)
% Try multiple naming patterns for per-mode summary CSVs.
% Returns numeric accuracy vector in [0,1]. Empty if not found.
    model = string(model); mode = string(mode);
    candidates = strings(0);

    % Preferred patterns with mode suffix
    candidates(end+1) = fullfile(resultsDir, sprintf('%s_accuracy_summary_%s.csv', model, mode));
    candidates(end+1) = fullfile(resultsDir, sprintf('%s_%s_accuracy_summary.csv', model, mode));
    % Legacy (no mode suffix) as a fallback
    candidates(end+1) = fullfile(resultsDir, sprintf('%s_accuracy_summary.csv', model));

    for i = 1:numel(candidates)
        f = candidates(i);
        if exist(f, 'file')
            try
                T = readtable(f);
                if any(strcmpi(T.Properties.VariableNames, 'accuracy'))
                    A = T.accuracy;
                    return;
                end
            catch
                % try robust reader
                T = local_read_table(f);
                if ismember('accuracy', T.Properties.VariableNames)
                    A = double(T.accuracy);
                    return;
                end
            end
        end
    end
    warning('Accuracy file for model %s (mode %s) not found in expected names. Skipping.', model, mode);
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
    % Coerce specific known numeric columns if present
    mustNum = {'seed','isTrain','index','y_true','y_pred','accuracy'};
    for i=1:numel(mustNum)
        n = mustNum{i};
        if ismember(n, T.Properties.VariableNames)
            T.(n) = double(str2double(T.(n)));
        end
    end
end

function save_fig(figH, ~, pngPath, pdfPath)
    if ~ishandle(figH) || ~strcmp(get(figH,'Type'),'figure')
        warning('Invalid figure handle. Skipping save.');
        return;
    end
    try
        set(figH, 'InvertHardcopy','off');
        set(figH, 'PaperPositionMode','auto');
        print(figH, '-dpng', '-r600', pngPath);
        print(figH, '-dpdf', '-painters', pdfPath);
    catch ME
        warning(ME.identifier, 'Failed to export figure with print: %s', ME.message);
        try
            saveas(figH, pngPath);
            saveas(figH, pdfPath);
        catch ME2
            warning(ME2.identifier, 'Fallback saveas also failed: %s', ME2.message);
        end
    end
end

