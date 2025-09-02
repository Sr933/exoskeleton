%% ================================================================
% Run LSTM using exoskeleton_library on preprocessed data.mat
% Mirrors the CNN pipeline: dynamic paths, 5 seeds, CSV outputs, headless
% ================================================================
clear; clc;

% ---------------- Resolve paths dynamically ----------------
scriptFullPath = mfilename('fullpath');
scriptDir = fileparts(scriptFullPath);
projectRoot = fileparts(scriptDir); % parent of "Machine learning"
matFile = fullfile(projectRoot, 'Preprocessed data', 'data.mat');
resultsDir = fullfile(projectRoot, 'results');
if ~exist(resultsDir, 'dir'); mkdir(resultsDir); end

% ---------------- Channel selection ----------------
% Choose which channels to load from the tables:
%   "all" (default) | "emg" | "imu"
channelMode = "imu";   % change to "emg" or "imu" as needed


% ---------------- Load -----------------
S = load(matFile);
assert(isfield(S,'data'), 'Expected variable ''data'' in MAT file.');
data = S.data;

% ---------------- Labels ---------------
label_map = exoskeleton_library.label_mapping();

% ---------------- Load raw as [C x T x N] ----------------
% Pass channel mode to loader
[Xc, y, meta] = exoskeleton_library.load_data_CNN(data, label_map, channelMode);

if isempty(Xc) || isempty(y)
    error('No valid data found for LSTM processing. Check the preprocessing step.');
end

% ---------------- Run multiple seeds and save accuracies ----------------
seeds = [1 2 3 4 5];
accs = zeros(numel(seeds),1);
allRows = table('Size',[0 9], 'VariableTypes', ...
    {'double','double','double','string','string','string','string','double','double'}, ...
    'VariableNames', {'seed','isTrain','index','folder','subject','part','file','y_true','y_pred'});

for si = 1:numel(seeds)
    seed = seeds(si);
    fprintf('\n=== Seed %d (LSTM) ===\n', seed);

    % Split (same format as CNN splitter)
    [Xtr, Xte, ytr, yte, idxTr, idxTe] = exoskeleton_library.split_data_CNN(Xc, y, seed);

    % Compress channels -> single channel by averaging (keeps length T)
    Xtr = flatten_channels_to_single(Xtr, "mean");     % much shorter sequences vs append
    Xte = flatten_channels_to_single(Xte, "mean");

    % Temporal downsampling to shorten sequences further (speed)
    dsTime = 1;  % try 10..25 depending on speed/accuracy trade-off
    Xtr = temporal_downsample_cells(Xtr, dsTime, "mean");
    Xte = temporal_downsample_cells(Xte, dsTime, "mean");

    % Sequence-to-label: categorical responses
    ytr_cat = toCategoricalLabels(ytr);
    yte_cat = toCategoricalLabels(yte);

    % Model: no time stride inside (keeps output length == input length)
    inpSize = size(Xtr{1},1);  % 1 after averaging channels
    layers  = exoskeleton_library.LSTM(inpSize);
    options = exoskeleton_library.LSTM_training_options();

    trainedNet = trainNetwork(Xtr, ytr_cat, layers, options);

    % Evaluate
    YPredTe = classify(trainedNet, Xte);
    catsTe = categories(yte_cat);
    YPredTe = categorical(YPredTe, catsTe, catsTe);
    accuracy = mean(YPredTe(:) == yte_cat(:));
    accs(si) = accuracy;
    fprintf('Seed %d - Test Accuracy (LSTM): %.2f%%\n', seed, accuracy*100);

    % Predict train to log details as well
    YPredTr = classify(trainedNet, Xtr);
    catsTr = categories(ytr_cat);
    YPredTr = categorical(YPredTr, catsTr, catsTr);
    ytr_idx       = double(categorical(ytr_cat, catsTr, catsTr));
    yte_idx       = double(categorical(yte_cat, catsTe, catsTe));
    ypred_tr_idx  = double(YPredTr);
    ypred_te_idx  = double(YPredTe);

    % Robust meta columns
    mFolderTr  = safe_meta(meta, idxTr, 'folder');
    mSubjectTr = safe_meta(meta, idxTr, 'subject');
    mPartTr    = safe_meta(meta, idxTr, 'part');
    mFileTr    = safe_meta(meta, idxTr, 'file');
    mFolderTe  = safe_meta(meta, idxTe, 'folder');
    mSubjectTe = safe_meta(meta, idxTe, 'subject');
    mPartTe    = safe_meta(meta, idxTe, 'part');
    mFileTe    = safe_meta(meta, idxTe, 'file');

    % Append rows: training set
    if ~isempty(idxTr)
        Ttr = table(repmat(seed,numel(idxTr),1), ones(numel(idxTr),1), idxTr(:), ...
            mFolderTr, mSubjectTr, mPartTr, mFileTr, ...
            ytr_idx(:), ypred_tr_idx(:), ...
            'VariableNames', {'seed','isTrain','index','folder','subject','part','file','y_true','y_pred'});
        allRows = [allRows; Ttr]; %#ok<AGROW>
    end
    % Append rows: test set
    if ~isempty(idxTe)
        Tte = table(repmat(seed,numel(idxTe),1), zeros(numel(idxTe),1), idxTe(:), ...
            mFolderTe, mSubjectTe, mPartTe, mFileTe, ...
            yte_idx(:), ypred_te_idx(:), ...
            'VariableNames', {'seed','isTrain','index','folder','subject','part','file','y_true','y_pred'});
        allRows = [allRows; Tte]; %#ok<AGROW>
    end
end

% Save summary CSV
Ts = table(seeds(:), accs(:), 'VariableNames', {'seed','accuracy'});
% Include channelmode
summaryCsv = fullfile(resultsDir, sprintf('lstm_accuracy_summary_%s.csv', channelMode));
try
    writetable(Ts, summaryCsv);
catch ME
    warning(ME.identifier, 'Failed to save LSTM summary CSV: %s', ME.message);
end

% Save detailed per-sample CSV (all seeds combined)
detailsCsv = fullfile(resultsDir, sprintf('lstm_predictions_all_seeds_%s.csv', channelMode));
try
    writetable(allRows, detailsCsv);
catch ME
    warning(ME.identifier, 'Failed to save LSTM details CSV: %s', ME.message);
end

function yc = toCategoricalLabels(y)
% Accepts: cell of one-hot vectors, numeric one-hot matrix, or categorical
    if iscategorical(y)
        yc = y; return;
    end
    if iscell(y)
        n = numel(y);
        idx = nan(n,1); K = 0;
        for i=1:n
            v = y{i}; v = v(:);
            [~, k] = max(v);
            idx(i) = k;
            K = max(K, numel(v));
        end
        yc = categorical(idx, 1:K, string(1:K));
        return;
    end
    if isnumeric(y)
        % Try KxN (columns are samples)
        if size(y,1) <= size(y,2)
            [~, idx] = max(y, [], 1); idx = idx(:);
            K = size(y,1);
        else
            % NxK (rows are samples)
            [~, idx] = max(y, [], 2); idx = idx(:);
            K = size(y,2);
        end
        yc = categorical(idx, 1:K, string(1:K));
        return;
    end
    error('Unsupported label type for conversion to categorical.');
end

function col = safe_meta(meta, idx, field)
% Return a string column for meta(idx).(field), robust to cells/numerics/empties.
    col = strings(numel(idx),1);
    for ii = 1:numel(idx)
        v = "";
        try
            v = meta(idx(ii)).(field);
        catch
            % leave as ""
        end
        col(ii) = to_string_scalar(v);
    end
end

function out = to_string_scalar(v)
% Convert any value to a string scalar safely.
    if isstring(v)
        if isscalar(v)
            out = v;
        else
            out = strjoin(v(:).', " ");
        end
    elseif ischar(v)
        out = string(v);
    elseif iscell(v)
        if isempty(v)
            out = "";
        else
            try
                out = to_string_scalar(v{1});
            catch
                out = "";
            end
        end
    elseif isnumeric(v) || islogical(v)
        out = string(v);
    else
        try
            out = string(v);
        catch
            out = "";
        end
    end
end

function [y_seq, y_seqlevel] = toSequenceLabels(Xcell, y, ds)
% Convert labels to per-timestep targets. If ds>1, use ceil(T/ds) length.
    if nargin < 3 || isempty(ds), ds = 1; end
    y_seqlevel = toCategoricalLabels(y);   % sequence-level categories
    N = numel(Xcell);
    y_seq = cell(N,1);
    for i = 1:N
        Ti = size(Xcell{i}, 2);
        Ti_out = ceil(Ti / ds);
        yi = y_seqlevel(i);
        y_seq{i} = repmat(yi, 1, Ti_out);  % categorical row vector
    end
end

function y_seqlevel = seq_majority(YPredSeq)
% Majority vote across timesteps per sequence.
% Input: cell of categorical row vectors; Output: 1xN categorical
    N = numel(YPredSeq);
    % Build global category set
    catsAll = categories(YPredSeq{1});
    for i = 2:N
        catsAll = union(catsAll, categories(YPredSeq{i}));
    end
    y_seqlevel = categorical(zeros(1,N), 1:numel(catsAll), catsAll);
    for i = 1:N
        yi = YPredSeq{i};
        % Ensure comparable categories
        yi = categorical(yi, catsAll, catsAll);
        % Mode across time
        try
            m = mode(yi, 2); % row vector -> mode scalar
        catch
            % Fallback if older MATLAB
            c = countcats(yi);
            [~, mx] = max(c);
            m = categorical(catsAll(mx), catsAll, catsAll);
        end
        y_seqlevel(i) = m;
    end
end

function Xds = temporal_downsample_cells(Xcell, ds, method)
% Downsample each [C x T] sequence along time by factor ds.
% method: "mean" (anti-alias) or "stride" (pick every ds-th sample)
    if nargin < 3, method = "mean"; end
    Xds = Xcell;
    for i = 1:numel(Xcell)
        Xi = Xcell{i};
        if isempty(Xi) || ~isnumeric(Xi)
            Xds{i} = Xi; continue;
        end
        [C, T] = size(Xi);
        if ds <= 1 || T < 2
            Xds{i} = Xi; continue;
        end
        switch lower(string(method))
            case "stride"
                idx = 1:ds:T;
                Xds{i} = Xi(:, idx);
            otherwise % "mean" pooling over non-overlapping blocks
                outT = floor(T / ds);
                if outT < 1
                    Xds{i} = Xi(:,1); continue;
                end
                Tfit = outT * ds;
                Xr = reshape(Xi(:,1:Tfit), C, ds, outT);
                Xm = squeeze(mean(Xr, 2));
                if isvector(Xm), Xm = reshape(Xm, C, outT); end
                Xds{i} = Xm;
        end
    end
end

function X1 = flatten_channels_to_single(Xcell, mode)
% Convert each [C x T] to [1 x T] (mean) or [1 x (T*C)] (append) or interleave.
% mode: "mean" (fast, recommended), "append", "interleave"
    if nargin < 2, mode = "mean"; end
    X1 = Xcell;
    for i = 1:numel(Xcell)
        Xi = Xcell{i};
        if isempty(Xi) || ~isnumeric(Xi)
            X1{i} = Xi; continue;
        end
        [C, T] = size(Xi);
        if C == 1
            X1{i} = Xi; continue;
        end
        switch lower(string(mode))
            case "append"
                X1{i} = reshape(Xi.', 1, []);           % [1 x (T*C)]
            case "interleave"
                X1{i} = reshape(Xi, 1, []);             % [1 x (C*T)]
            otherwise % "mean" across channels -> [1 x T]
                X1{i} = mean(Xi, 1);
        end
    end
end
