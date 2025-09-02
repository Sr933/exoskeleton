%% ================================================================
% Random baseline analogous to SVM baseline
% - Dynamic paths
% - 5 seeds
% - Uses same data loading as other baselines
% - Predicts random labels using training set class distribution
% - Saves summary and detailed CSVs
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
% "all" (default) | "emg" | "imu"
channelMode = "emg";   % change to "emg" or "imu" as needed

% ---------------- Load -----------------
S = load(matFile);
assert(isfield(S,'data'), 'Expected variable ''data'' in MAT file.');
data = S.data;

% ---------------- Labels ---------------
label_map = exoskeleton_library.label_mapping();

% ---------------- Load raw as cell + meta ----------------
[Xc, y, meta] = exoskeleton_library.load_data_CNN(data, label_map, channelMode);
assert(~isempty(Xc) && ~isempty(y), 'No valid data found for random baseline.');
N = numel(Xc);

% ---------------- Multi-seed evaluation ----------------
seeds = [1 2 3 4 5];
accs = zeros(numel(seeds),1);
allRows = table('Size',[0 9], 'VariableTypes', ...
    {'double','double','double','string','string','string','string','double','double'}, ...
    'VariableNames', {'seed','isTrain','index','folder','subject','part','file','y_true','y_pred'});

% One-hot to class index helper
onehot_to_idx = @(cellvec) cellfun(@local_onehot_to_idx, cellvec);
% Determine K safely from first non-empty one-hot vector
nonEmptyIdx = find(cellfun(@(a) ~isempty(a) && isnumeric(a), y), 1, 'first');
assert(~isempty(nonEmptyIdx), 'Empty labels.');
K = numel(y{nonEmptyIdx});

% Safe meta extraction upfront
[folders, subjects, parts, files] = local_meta_fields(meta);

for si = 1:numel(seeds)
    seed = seeds(si);
    rng(seed);
    fprintf('\n=== Seed %d (Random baseline) ===\n', seed);

    % Split indices (40% test to mirror SVM baseline)
    [idxTr, idxTe] = local_partition_indices(N, seed, 0.4);
    ytr_cells = y(idxTr);
    yte_cells = y(idxTe);
    ytr_idx = onehot_to_idx(ytr_cells);
    yte_idx = onehot_to_idx(yte_cells);

    % Ensure column vectors
    ytr_idx = ytr_idx(:);
    yte_idx = yte_idx(:);

    % Empirical class probabilities from training set (fallback to uniform)
    validTr = ~isnan(ytr_idx);
    subs = ytr_idx(validTr);
    subs = subs(:);
    subs = subs(subs >= 1 & subs <= K);
    if isempty(subs)
        counts = zeros(K,1);
    else
        counts = accumarray(subs, 1, [K 1]);
    end
    if sum(counts) == 0
        probs = ones(K,1) / K;
    else
        probs = counts / sum(counts);
        if any(~isfinite(probs)) || any(probs < 0)
            probs = ones(K,1) / K;
        end
    end

    % Random predictions for train and test using probs
    ypred_tr_idx = randsample(K, numel(ytr_idx), true, probs);
    ypred_te_idx = randsample(K, numel(yte_idx), true, probs);

    % Accuracy on test (ignore NaN true labels if any)
    validTe = ~isnan(yte_idx);
    if any(validTe)
        acc = sum(ypred_te_idx(validTe) == yte_idx(validTe)) / nnz(validTe);
    else
        acc = NaN;
    end
    accs(si) = acc;
    fprintf('Seed %d - Test Accuracy (Random): %.2f%%\n', seed, acc*100);

    % Append rows: training set
    if ~isempty(idxTr)
        Ttr = table(repmat(seed,numel(idxTr),1), ones(numel(idxTr),1), idxTr(:), ...
            folders(idxTr), subjects(idxTr), parts(idxTr), files(idxTr), ...
            double(ytr_idx(:)), double(ypred_tr_idx(:)), ...
            'VariableNames', {'seed','isTrain','index','folder','subject','part','file','y_true','y_pred'});
        allRows = [allRows; Ttr]; %#ok<AGROW>
    end

    % Append rows: test set
    if ~isempty(idxTe)
        Tte = table(repmat(seed,numel(idxTe),1), zeros(numel(idxTe),1), idxTe(:), ...
            folders(idxTe), subjects(idxTe), parts(idxTe), files(idxTe), ...
            double(yte_idx(:)), double(ypred_te_idx(:)), ...
            'VariableNames', {'seed','isTrain','index','folder','subject','part','file','y_true','y_pred'});
        allRows = [allRows; Tte]; %#ok<AGROW>
    end
end

% Save summary CSV
Ts = table(seeds(:), accs(:), 'VariableNames', {'seed','accuracy'});
summaryCsv = fullfile(resultsDir, sprintf('random_accuracy_summary_%s.csv', channelMode));
try
    writetable(Ts, summaryCsv);
catch ME
    warning(ME.identifier, 'Failed to save Random summary CSV: %s', ME.message);
end

% Save detailed per-sample CSV (all seeds combined)
detailsCsv = fullfile(resultsDir, sprintf('random_predictions_all_seeds_%s.csv', channelMode));
try
    writetable(allRows, detailsCsv);
catch ME
    warning(ME.identifier, 'Failed to save Random details CSV: %s', ME.message);
end

%% ---------------- Local helpers ----------------
function k = local_onehot_to_idx(v)
% Convert a one-hot row vector to class index (1..K). NaN if invalid.
    if isempty(v) || ~isnumeric(v)
        k = NaN; return;
    end
    v = v(:).';
    if all(~isfinite(v)) || all(v==0)
        k = NaN; return;
    end
    [~, k] = max(v,[],2);
end

function [idxTr, idxTe] = local_partition_indices(N, seed, testFrac)
% Deterministic random split by seed
    if nargin < 3 || isempty(testFrac), testFrac = 0.4; end
    rng(seed);
    p = randperm(N);
    nTe = max(1, round(testFrac * N));
    te = p(1:nTe);
    tr = p(nTe+1:end);
    idxTr = tr(:); idxTe = te(:);
end

function [folders, subjects, parts, files] = local_meta_fields(meta)
% Safely extract meta string arrays; fill missing with ""
    N = numel(meta);
    folders  = strings(N,1);
    subjects = strings(N,1);
    parts    = strings(N,1);
    files    = strings(N,1);
    for i=1:N
        if isfield(meta, 'folder')  && ~isempty(meta(i).folder),  folders(i)  = string(meta(i).folder);  end
        if isfield(meta, 'subject') && ~isempty(meta(i).subject), subjects(i) = string(meta(i).subject); end
        if isfield(meta, 'part')    && ~isempty(meta(i).part),    parts(i)    = string(meta(i).part);    end
        if isfield(meta, 'file')    && ~isempty(meta(i).file),    files(i)    = string(meta(i).file);    end
    end
end

