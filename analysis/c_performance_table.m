%% ================================================================
% Analysis: CNN performance table across seeds (Accuracy, Precision, Recall, F1)
% - Reads results/cnn_predictions_all_seeds_<channelMode>.csv
%   where channelMode in {"all","left_leg","right_leg"}
% - Uses test-only rows (isTrain==0)
% - Computes per-seed macro Precision/Recall/F1 and Accuracy, then mean/std
% - Saves results/cnn_metrics_summary_<channelMode>.csv
%   Also writes results/cnn_metrics_summary.csv for mode=="all" (compat)
%   Also writes results/cnn_metrics_summary_all_modes.csv (aggregated)
% ================================================================
clear; clc;

% --------------- Paths ----------------
scriptFullPath = mfilename('fullpath');
scriptDir = fileparts(scriptFullPath);
projectRoot = fileparts(scriptDir);
resultsDir = fullfile(projectRoot, 'results');

% Process multiple channel modes
channelModes = ["all","left_leg","right_leg"];
processed = false;
aggRows = table('Size',[0 4], ...
    'VariableTypes', {'string','string','double','double'}, ...
    'VariableNames', {'ChannelMode','Metric','Mean','Std'});

for cm = channelModes
    predFile = fullfile(resultsDir, sprintf('cnn_predictions_all_seeds_%s.csv', cm));
    if exist(predFile,'file') ~= 2
        warning('Predictions CSV not found for channelMode=%s: %s', cm, predFile);
        continue;
    end

    T = local_read_table(predFile);
    req = {'seed','isTrain','y_true','y_pred'};
    assert(all(ismember(req, T.Properties.VariableNames)), ...
        'Predictions CSV missing required columns for channelMode=%s.', cm);

    % Filter test-only rows
    T = T(T.isTrain==0, :);
    if isempty(T)
        warning('No test rows found in predictions CSV for channelMode=%s.', cm);
        continue;
    end

    % Determine class set
    K = max([max(T.y_true), max(T.y_pred)]);
    order = 1:K;

    % Group by seed
    seeds = unique(T.seed);
    acc  = nan(numel(seeds),1);
    prec = nan(numel(seeds),1);
    rec  = nan(numel(seeds),1);
    f1   = nan(numel(seeds),1);

    for i = 1:numel(seeds)
        s = seeds(i);
        Ti = T(T.seed==s, :);
        if isempty(Ti), continue; end
        C = confusionmat(Ti.y_true, Ti.y_pred, 'Order', order);
        [acc(i), prec(i), rec(i), f1(i)] = metrics_from_confusion(C);
    end

    % Summary (mean and std across seeds)
    metrics = ["Accuracy"; "Precision (macro)"; "Recall (macro)"; "F1 (macro)"];
    means   = [mean(acc,'omitnan'); mean(prec,'omitnan'); mean(rec,'omitnan'); mean(f1,'omitnan')];
    stds    = [ std(acc,'omitnan');  std(prec,'omitnan');  std(rec,'omitnan');  std(f1,'omitnan')];

    summary = table(metrics, means, stds, ...
        'VariableNames', {'Metric','Mean','Std'});

    % Save per-mode summary
    outFile = fullfile(resultsDir, sprintf('cnn_metrics_summary_%s.csv', cm));
    try
        writetable(summary, outFile);
        fprintf('Saved %s summary to %s\n', cm, outFile);
    catch ME
        warning(ME.identifier, 'Failed to save summary CSV (%s): %s', cm, ME.message);
    end

    % Back-compat: also write generic name for "all"
    if cm == "all"
        outCompat = fullfile(resultsDir, 'cnn_metrics_summary.csv');
        try
            writetable(summary, outCompat);
            fprintf('Saved back-compat summary to %s\n', outCompat);
        catch ME
            warning(ME.identifier, 'Failed to save back-compat CSV: %s', ME.message);
        end
    end

    % Aggregate rows for combined file
    aggRows = [aggRows; ...
        table(repmat(string(cm),numel(metrics),1), metrics, means, stds, ...
              'VariableNames', {'ChannelMode','Metric','Mean','Std'})]; %#ok<AGROW>

    processed = true;
end

% Save aggregated summary across modes
if processed && ~isempty(aggRows)
    outAgg = fullfile(resultsDir, 'cnn_metrics_summary_all_modes.csv');
    try
        writetable(aggRows, outAgg);
        fprintf('Saved aggregated summary to %s\n', outAgg);
    catch ME
        warning(ME.identifier, 'Failed to save aggregated summary CSV: %s', ME.message);
    end
else
    warning('No summaries generated. Ensure prediction CSVs exist in %s', resultsDir);
end

% ---------------- Local helpers ----------------
function [acc, prec_macro, rec_macro, f1_macro] = metrics_from_confusion(C)
% C: KxK confusion matrix, rows=true, cols=pred
tp = diag(C);
fp = sum(C,1).' - tp; % predicted positives minus tp
fn = sum(C,2) - tp;   % actual positives minus tp
tn = sum(C(:)) - tp - fp - fn; %#ok<NASGU>

% Per-class metrics
prec = tp ./ (tp + fp);
rec  = tp ./ (tp + fn);
f1c  = 2 .* (prec .* rec) ./ (prec + rec);

% Handle NaNs (e.g., 0/0): ignore those classes in macro average
prec_macro = mean(prec(~isnan(prec)));
rec_macro  = mean(rec(~isnan(rec)));
f1_macro   = mean(f1c(~isnan(f1c)));

% Accuracy
acc = sum(tp) / sum(C(:));
end

function T = local_read_table(file)
    % Robust CSV reader: prefer readtable(FileType='text'), then manual parse
    try
        T = readtable(file, 'PreserveVariableNames', true, 'FileType','text');
        % Coerce numeric types if needed
        mustNum = {'seed','isTrain','y_true','y_pred'};
        for i=1:numel(mustNum)
            n = mustNum{i};
            if ismember(n, T.Properties.VariableNames)
                T.(n) = double(T.(n));
            end
        end
        return;
    catch
        % fall through
    end
    % Manual CSV parse
    fid = fopen(file, 'r');
    if fid == -1, error('Unable to open file %s', file); end
    headerLine = fgetl(fid);
    if ~ischar(headerLine)
        fclose(fid); error('File %s appears empty or invalid.', file);
    end
    headers = strsplit(headerLine, {',',';','\t'});
    numCols = numel(headers);
    fmt = repmat('%s', 1, numCols);
    C = textscan(fid, fmt, 'Delimiter', {',',';','\t'}, 'EndOfLine','\n');
    fclose(fid);
    T = table();
    for i=1:numCols
        col = C{i};
        T.(matlab.lang.makeValidName(strtrim(headers{i}))) = string(col);
    end
    mustNum = {'seed','isTrain','y_true','y_pred'};
    for i=1:numel(mustNum)
        n = mustNum{i};
        if ismember(n, T.Properties.VariableNames)
            T.(n) = double(str2double(T.(n)));
        end
    end
end

