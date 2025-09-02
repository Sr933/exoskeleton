%% ================================================================
% Run SVM baseline on preprocessed data.mat using exoskeleton_library helpers
% Mirrors CNN/LSTM: dynamic paths, 5 seeds, CSV outputs (summary + details)
% ================================================================
clear; clc;

% ---------------- Resolve paths dynamically ----------------
scriptFullPath = mfilename('fullpath');
scriptDir = fileparts(scriptFullPath);
projectRoot = fileparts(scriptDir); % parent of "src"
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

% ---------------- Load raw and build features/labels ----------------
[Xc, y, meta] = exoskeleton_library.load_data_CNN(data, label_map, channelMode);
assert(~isempty(Xc) && ~isempty(y), 'No valid data found for SVM processing.');
% Build matrix by padding to max T and flattening raw signals (no features)
X     = local_flatten_cells(Xc);           % [N x (C*maxT)] raw flattened
y_cat = local_to_categorical(y);           % categorical labels
N     = numel(Xc);

% ---------------- Multi-seed evaluation ----------------
seeds = [1 2 3 4 5];
accs = zeros(numel(seeds),1);
allRows = table('Size',[0 9], 'VariableTypes', ...
    {'double','double','double','string','string','string','string','double','double'}, ...
    'VariableNames', {'seed','isTrain','index','folder','subject','part','file','y_true','y_pred'});

for si = 1:numel(seeds)
    seed = seeds(si);
    fprintf('\n=== Seed %d (SVM) ===\n', seed);

    % Stratified split by label (40% test)
    rng(seed);
    c = cvpartition(y_cat, 'Holdout', 0.4);
    idxTr = find(training(c));
    idxTe = find(test(c));
    % Safety: ensure at least 1 train and 1 test
    if isempty(idxTr) || isempty(idxTe)
        % fallback deterministic split
        [idxTr, idxTe] = local_partition_indices(N, seed, 0.4);
    end
    Xtr = X(idxTr, :); Xte = X(idxTe, :);
    ytr = y_cat(idxTr); yte = y_cat(idxTe);
    % Ensure training has >=2 classes
    if numel(categories(ytr)) < 2 && numel(unique(ytr)) < 2
        warning('Seed %d produced 1-class train set; resplitting with new seed.', seed);
        [idxTr, idxTe] = local_partition_indices(N, seed+100, 0.4);
        Xtr = X(idxTr, :); Xte = X(idxTe, :);
        ytr = y_cat(idxTr); yte = y_cat(idxTe);
    end

    % Train weaker linear SVM ECOC (one-vs-all), no standardization, low C
    t = templateSVM('KernelFunction','linear');
    Mdl = fitcecoc(Xtr, ytr, 'Learners', t, 'Coding','onevsall');

    % Predict
    ypred_tr = predict(Mdl, Xtr);
    ypred_te = predict(Mdl, Xte);

    % Accuracy on test
    acc = mean(ypred_te == yte);
    accs(si) = acc;
    fprintf('Seed %d - Test Accuracy (SVM): %.2f%%\n', seed, acc*100);

    % Convert to numeric indices for CSV (1..K codes)
    ytr_idx = double(ytr);
    yte_idx = double(yte);
    ypred_tr_idx = double(ypred_tr);
    ypred_te_idx = double(ypred_te);

    % Safe meta extraction
    [folders, subjects, parts, files] = local_meta_fields(meta);

    % Append rows: training set
    if ~isempty(idxTr)
        Ttr = table(repmat(seed,numel(idxTr),1), ones(numel(idxTr),1), idxTr(:), ...
            folders(idxTr), subjects(idxTr), parts(idxTr), files(idxTr), ...
            ytr_idx(:), ypred_tr_idx(:), ...
            'VariableNames', {'seed','isTrain','index','folder','subject','part','file','y_true','y_pred'});
        allRows = [allRows; Ttr]; %#ok<AGROW>
    end

    % Append rows: test set
    if ~isempty(idxTe)
        Tte = table(repmat(seed,numel(idxTe),1), zeros(numel(idxTe),1), idxTe(:), ...
            folders(idxTe), subjects(idxTe), parts(idxTe), files(idxTe), ...
            yte_idx(:), ypred_te_idx(:), ...
            'VariableNames', {'seed','isTrain','index','folder','subject','part','file','y_true','y_pred'});
        allRows = [allRows; Tte]; %#ok<AGROW>
    end
end

% Save summary CSV
Ts = table(seeds(:), accs(:), 'VariableNames', {'seed','accuracy'});
summaryCsv = fullfile(resultsDir, sprintf('svm_accuracy_summary_%s.csv', channelMode));
try
    writetable(Ts, summaryCsv);
catch ME
    warning(ME.identifier, 'Failed to save SVM summary CSV: %s', ME.message);
end

% Save detailed per-sample CSV (all seeds combined)
detailsCsv = fullfile(resultsDir, sprintf('svm_predictions_all_seeds_%s.csv', channelMode));
try
    writetable(allRows, detailsCsv);
catch ME
    warning(ME.identifier, 'Failed to save SVM details CSV: %s', ME.message);
end



function ycat = local_to_categorical(y)
% Accepts: cell of one-hot vectors, numeric indices, or categorical
    if iscell(y)
        % Assume one-hot or numeric per-cell
        idx = zeros(numel(y),1);
        K = 0;
        for ii=1:numel(y)
            yi = y{ii};
            if isnumeric(yi)
                yi = yi(:).';
                [~, k] = max(yi);
                if isempty(k) || all(yi==0), k = 1; end
                idx(ii) = k; K = max(K, numel(yi));
            else
                error('Unsupported label cell element type.');
            end
        end
        ycat = categorical(idx, 1:max(idx));
    elseif iscategorical(y)
        ycat = y;
    elseif isnumeric(y)
        ycat = categorical(y);
    else
        error('Unsupported label type.');
    end
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

function X = local_flatten_cells(Xcells)
% Zero-pad each [C x T] to max T and flatten to 1 x (C*T).
    N = numel(Xcells);
    assert(N>0, 'Empty input.');
    % Ensure consistent channel count
    C = size(Xcells{1},1);
    Ts = zeros(N,1);
    for i=1:N
        Xi = Xcells{i};
        if size(Xi,1) ~= C
            error('Inconsistent channel count at sample %d (got %d, expected %d).', i, size(Xi,1), C);
        end
        Ts(i) = size(Xi,2);
    end
    Tm = max(Ts);
    X = zeros(N, C*Tm);
    for i=1:N
        Xi = double(Xcells{i});
        Ti = size(Xi,2);
        if Ti < Tm
            Xi = [Xi, zeros(C, Tm-Ti)]; %#ok<AGROW> % pad zeros to the right
        elseif Ti > Tm
            Xi = Xi(:,1:Tm); % truncate if any sample longer (shouldn't happen)
        end
        X(i,:) = Xi(:).';
    end
end

