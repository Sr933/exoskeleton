%% ================================================================
% Run CNN using exoskeleton_library on preprocessed data.mat
% Train with trainNetwork: predictors are N-by-1 cell arrays [C x T]
% ================================================================
clear; clc;

% ---------------- Paths ----------------
scriptFullPath = mfilename('fullpath');
scriptDir = fileparts(scriptFullPath);
projectRoot = fileparts(scriptDir); % parent folder
matFile = fullfile(projectRoot, 'Preprocessed data', 'data.mat');
resultsDir = fullfile(projectRoot, 'results');
if ~exist(resultsDir, 'dir'); mkdir(resultsDir); end

% ---------------- Channel selection ----------------
% Choose which channels to load from the tables:
%   "all" | "emg" | "imu" | "right_leg" | "left_leg"
%   right_leg = IMU 1 & 3 + EMG 1–4
%   left_leg  = IMU 2     + EMG 5–8
channelMode = "right_leg";   % set to "left_leg" to compare the other group


% ---------------- Load -----------------
S = load(matFile);
assert(isfield(S,'data'), 'Expected variable ''data'' in MAT file.');
data = S.data;

% ---------------- Labels ---------------
label_map = exoskeleton_library.label_mapping();

% ---------------- Load raw as [C x T x N] from library ----------------
[Xc, y, meta] = exoskeleton_library.load_data_CNN(data, label_map, channelMode);

% ---- Diagnostics: check channel selection / data size ----
fprintf('\n[Diagnostics] channelMode = %s\n', string(channelMode));
N  = numel(Xc);
Cs = arrayfun(@(i) size(Xc{i},1), 1:N);
Ts = arrayfun(@(i) size(Xc{i},2), 1:N);
uc = unique(Cs);
fprintf('Samples N = %d | Channels C = %s | T (min/median/max) = %d / %d / %d\n', ...
    N, mat2str(uc), min(Ts), round(median(Ts)), max(Ts));
if numel(uc) > 1
    warning('Inconsistent channel count across samples: %s', mat2str(uc));
end
% Try to print channel names from meta (if provided by the library)
chanNames = {};
candFields = {'channels','channelNames','varNames','vars','selectedChannels','used'};
for k = 1:numel(candFields)
    f = candFields{k};
    if ~isempty(meta) && isfield(meta(1), f) && ~isempty(meta(1).(f))
        v = meta(1).(f);
        if isstring(v)
            chanNames = cellstr(v);
        elseif iscell(v) && all(cellfun(@(x) ischar(x) || isstring(x), v))
            chanNames = cellstr(string(v));
        end
        if ~isempty(chanNames), break; end
    end
end
if ~isempty(chanNames)
    fprintf('Selected channels (%d): %s\n', numel(chanNames), strjoin(chanNames, ', '));
else
    fprintf('Selected channel names not available in meta.\n');
end
% ---------------------------------------------------------

if isempty(Xc) || isempty(y)
    error('No valid data found for CNN processing. Check the preprocessing step.');
end

% ---------------- Run multiple seeds and save accuracies ----------------
seeds = [1 2 3 4 5];
accs = zeros(numel(seeds),1);
allRows = table('Size',[0 9], 'VariableTypes', ...
    {'double','double','double','string','string','string','string','double','double'}, ...
    'VariableNames', {'seed','isTrain','index','folder','subject','part','file','y_true','y_pred'});

for si = 1:numel(seeds)
    seed = seeds(si);
    fprintf('\n=== Seed %d ===\n', seed);

    % Split
    [Xtr, Xte, ytr, yte, idxTr, idxTe] = exoskeleton_library.split_data_CNN(Xc, y, seed);

    % Convert one-hot cells -> categorical (required by trainNetwork)
    if iscell(ytr)
        [ytr, K] = onehot_cells_to_categorical(ytr);   % ytr categorical
        yte      = onehot_cells_to_categorical(yte, K);% yte categorical with same classes
    end

    % Model & options
    inpSize = size(Xtr{1},1);
    layers  = exoskeleton_library.CNN(inpSize);
    options = exoskeleton_library.CNN_training_options(); % headless

    % Train
    trainedNet = trainNetwork(Xtr, ytr, layers, options);

    % Evaluate
    YPred = classify(trainedNet, Xte);
    accuracy = sum(YPred(:) == yte(:)) / numel(yte);
    accs(si) = accuracy;
    fprintf('Seed %d - Test Accuracy: %.2f%%\n', seed, accuracy*100);

    % Collect detailed rows for train set and test set
    YPredTr = classify(trainedNet, Xtr);
    ytr_idx = double(ytr);
    yte_idx = double(yte);
    ypred_tr_idx = double(YPredTr);
    ypred_te_idx = double(YPred);

    % Robust meta columns (avoid string/horzcat size issues)
    mFolderTr  = safe_meta(meta, idxTr, 'folder');
    mSubjectTr = safe_meta(meta, idxTr, 'subject');
    mPartTr    = safe_meta(meta, idxTr, 'part');
    mFileTr    = safe_meta(meta, idxTr, 'file');
    mFolderTe  = safe_meta(meta, idxTe, 'folder');
    mSubjectTe = safe_meta(meta, idxTe, 'subject');
    mPartTe    = safe_meta(meta, idxTe, 'part');
    mFileTe    = safe_meta(meta, idxTe, 'file');

    if ~isempty(idxTr)
        Ttr = table(repmat(seed,numel(idxTr),1), ones(numel(idxTr),1), idxTr(:), ...
            mFolderTr, mSubjectTr, mPartTr, mFileTr, ...
            ytr_idx(:), ypred_tr_idx(:), ...
            'VariableNames', {'seed','isTrain','index','folder','subject','part','file','y_true','y_pred'});
        allRows = [allRows; Ttr]; %#ok<AGROW>
    end
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
% Include channelmode in filename
summaryCsv = fullfile(resultsDir, sprintf('cnn_accuracy_summary_%s.csv', channelMode));
try
    writetable(Ts, summaryCsv);
catch ME
    warning(ME.identifier, 'Failed to save summary CSV: %s', ME.message);
end

% Save detailed per-sample CSV (all seeds combined)
detailsCsv = fullfile(resultsDir, sprintf('cnn_predictions_all_seeds_%s.csv', channelMode));
try
    writetable(allRows, detailsCsv);
catch ME
    warning(ME.identifier, 'Failed to save details CSV: %s', ME.message);
end

%% ----------------- Helper (labels pretty-print) -----------------
function [names, order] = localClassNamesFromMapping(mapping)
    if isa(mapping, 'containers.Map')
        k = mapping.keys; v = mapping.values;
        [order, idx] = sort(cell2mat(v)); names = string(k(idx));
    elseif isstruct(mapping)
        order = [mapping.order]; names = string({mapping.name});
        f = fieldnames(mapping); vals = zeros(numel(f),1);
        for i=1:numel(f), vals(i) = mapping.(f{i}); end
        [order, idx] = sort(vals); names = string(f(idx));
    elseif iscell(mapping) || isstring(mapping)
        names = string(mapping); order = (1:numel(names)).';
    else
        names = "Class " + (1:max(cell2mat(values(mapping))));
        order = (1:numel(names)).';
    end
end

function [yc, K] = onehot_cells_to_categorical(ycells, K)
% Convert 1xK one-hot cells to categorical labels (1..K).
    n = numel(ycells);
    idx = zeros(n,1);
    Kdet = 0;
    for i = 1:n
        oh = ycells{i};
        if isempty(oh)
            idx(i) = NaN;
        else
            oh = oh(:);
            [~, k] = max(oh);
            idx(i) = k;
            Kdet = max(Kdet, numel(oh));
        end
    end
    if nargin < 2 || isempty(K), K = Kdet; end
    yc = categorical(idx, 1:K, string(1:K));
end

function col = safe_meta(meta, idx, field)
% Return a string column for meta(idx).(field), robust to cells/numerics/empties.
    col = strings(numel(idx),1);
    for ii = 1:numel(idx)
        v = [];
        try, v = meta(idx(ii)).(field); catch, end
        if isstring(v)
            col(ii) = strjoin(v(:).', " ");
        elseif ischar(v)
            col(ii) = string(v);
        elseif iscell(v)
            if isempty(v)
                col(ii) = "";
            else
                w = v{1};
                try
                    if isstring(w) || ischar(w)
                        col(ii) = string(w);
                    elseif isnumeric(w) || islogical(w)
                        col(ii) = string(w);
                    else
                        col(ii) = string(evalc('disp(w)'));
                    end
                catch
                    col(ii) = "";
                end
            end
        elseif isnumeric(v) || islogical(v)
            col(ii) = string(v);
        else
            try, col(ii) = string(v); catch, col(ii) = ""; end
        end
    end
end
