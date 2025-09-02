%% ================================================================
% Transfer Learning runner (CNN + trainnet fine-tuning)
% - Dynamic paths
% - One seed
% - For each subject as target:
%   * Pretrain on all other subjects
%   * Fine-tune on target: 10 samples/class or 10% if class has <10 samples
%   * Evaluate on remaining target samples
% - Save summary and detailed CSVs in results/
% ================================================================
clear; clc;

% ---------------- Resolve paths dynamically ----------------
scriptFullPath = mfilename('fullpath');
scriptDir = fileparts(scriptFullPath);
projectRoot = fileparts(scriptDir); % parent of "Machine learning"
matFile = fullfile(projectRoot, 'Preprocessed data', 'data.mat');
resultsDir = fullfile(projectRoot, 'results');
if ~exist(resultsDir, 'dir'); mkdir(resultsDir); end

% ---------------- Config ----------------
seed = 1;
rng(seed);

% ---------------- Load -----------------
S = load(matFile);
assert(isfield(S,'data'), 'Expected variable ''data'' in MAT file.');
data = S.data;

% ---------------- Labels ---------------
label_map = exoskeleton_library.label_mapping();
[Xc, y, meta] = exoskeleton_library.load_data_CNN(data, label_map);
assert(~isempty(Xc) && ~isempty(y), 'No valid data found.');

% Align lengths (avoid out-of-bounds)
Nall = [size(Xc,3), numel(y), numel(meta)];
N = min(Nall);
if any(Nall ~= N)
    warning('Truncating to common length N=%d (Xc=%d, y=%d, meta=%d).', N, Nall(1), Nall(2), Nall(3));
    Xc   = Xc(:,:,1:N);
    y    = y(1:N);
    meta = meta(1:N);
end

% Prepare helper label arrays (K x N one-hot for trainnet)
K = numel(y{1});
y_mat_all = zeros(K, N);
for n = 1:N
    vi = y{n}; if isrow(vi), vi = vi(:); end
    assert(numel(vi) == K, 'Label length mismatch at sample %d.', n);
    y_mat_all(:, n) = vi;
end
subjects = local_meta_col(meta, 'subject');    % 1 x N string (robust)
uSubjects = unique(subjects(subjects ~= ""), 'stable');

% Outputs (now includes acc_ft_only for "no pretraining")
sumRows = table('Size',[0 5], ...
    'VariableTypes', {'string','double','double','double','double'}, ...
    'VariableNames', {'subject','acc_fine_tuned','acc_pretrained_only','acc_ft_only','nFineTune'});
allRows = table('Size',[0 11], 'VariableTypes', ...
    {'string','double','double','double','string','string','string','string','double','double','string'}, ...
    'VariableNames', {'subject','seed','isTrain','index','folder','subjectStr','part','file','y_true','y_pred','model'});

% ---------------- Loop over each subject as target ----------------
for si = 1:numel(uSubjects)
    tgt = uSubjects(si);
    fprintf('\n=== Transfer Learning for subject %s ===\n', tgt);

    idxTgt = find(subjects == tgt);
    idxSrc = setdiff(1:N, idxTgt);
    idxTgt = idxTgt(idxTgt>=1 & idxTgt<=N);
    idxSrc = idxSrc(idxSrc>=1 & idxSrc<=N);

    % Datasets
    X_src = Xc(:,:,idxSrc);      % [C x T x Bsrc]
    Y_src = y_mat_all(:, idxSrc);% [K x Bsrc]
    X_tgt = Xc(:,:,idxTgt);
    Y_tgt = y_mat_all(:, idxTgt);

    % Split target into fine-tune (few per class) and test remainder
    Kt = size(Y_tgt,1);
    [~, tgt_idx_vec] = max(Y_tgt, [], 1);  % class index per target sample
    idxFt_rel = [];
    for c = 1:Kt
        rel = find(tgt_idx_vec == c);
        nClass = numel(rel);
        if nClass == 0, continue; end
        nPick = min(10, max(1, ceil(0.10 * nClass)));
        sel = rel(randperm(nClass, min(nPick, nClass)));
        idxFt_rel = [idxFt_rel, sel]; %#ok<AGROW>
    end
    idxFt_rel = unique(idxFt_rel, 'stable');
    idxFt = idxTgt(idxFt_rel);
    idxTe = setdiff(idxTgt, idxFt);

    % Build unchanged model
    inpSize = size(X_src,1);
    layers = exoskeleton_library.CNN(inpSize);

    % Pretrain on source subjects (RAM-safe chunks)
    y_src_cat = onehot_to_categorical(Y_src);
    net = pretrain_cnn_in_chunks(X_src, y_src_cat, layers);

    % Evaluate pretrained-only on target test (before fine-tune)
    acc_pre = NaN; acc_ft = NaN; acc_ftonly = NaN;
    if ~isempty(idxTe)
        X_te_cell = to_cellseq(Xc(:,:,idxTe));
        Y_te = y_mat_all(:, idxTe);
        ytrue_idx_pre = onehot_argmax(Y_te);
        ypred_pre_cat = classify(net, X_te_cell);
        I_pre = double(ypred_pre_cat);
        acc_pre = mean(I_pre(:) == ytrue_idx_pre(:));

        % Log pretrained-only test rows
        mFolderTe  = local_meta_col(meta(idxTe), 'folder').';
        mSubjectTe = local_meta_col(meta(idxTe), 'subject').';
        mPartTe    = local_meta_col(meta(idxTe), 'part').';
        mFileTe    = local_meta_col(meta(idxTe), 'file').';
        Tpre = table(repmat(tgt,numel(idxTe),1), repmat(seed,numel(idxTe),1), zeros(numel(idxTe),1), idxTe(:), ...
            mFolderTe, mSubjectTe, mPartTe, mFileTe, ...
            ytrue_idx_pre(:), I_pre(:), repmat("pretrained_only", numel(idxTe),1), ...
            'VariableNames', {'subject','seed','isTrain','index','folder','subjectStr','part','file','y_true','y_pred','model'});
        allRows = [allRows; Tpre]; %#ok<AGROW>
    end

    % "No pretraining": train from scratch on the fine-tune subset only
    nFt = numel(idxFt);
    if nFt > 0
        X_ft_cell = to_cellseq(Xc(:,:,idxFt));
        y_ft_cat  = onehot_to_categorical(y_mat_all(:, idxFt));

        opts_np = trainingOptions('adam', ...
            'MaxEpochs', 15, ...
            'MiniBatchSize', 32, ...
            'InitialLearnRate', 1e-3, ...
            'Shuffle', 'every-epoch', ...
            'ExecutionEnvironment', 'auto', ...
            'Verbose', false, 'Plots', 'none', ...
            'OutputNetwork', 'last-iteration');
        % Train scratch model (same architecture, no pretraining)
        net_np = trainNetwork(X_ft_cell, y_ft_cat, exoskeleton_library.CNN(size(Xc,1)), opts_np);

        if ~isempty(idxTe)
            X_te_cell = to_cellseq(Xc(:,:,idxTe));
            Y_te = y_mat_all(:, idxTe);
            ytrue_idx_np = onehot_argmax(Y_te);
            ypred_np_cat = classify(net_np, X_te_cell);
            I_np = double(ypred_np_cat);
            acc_ftonly = mean(I_np(:) == ytrue_idx_np(:));

            % Log no-pretraining test rows
            mFolderTe  = local_meta_col(meta(idxTe), 'folder').';
            mSubjectTe = local_meta_col(meta(idxTe), 'subject').';
            mPartTe    = local_meta_col(meta(idxTe), 'part').';
            mFileTe    = local_meta_col(meta(idxTe), 'file').';
            Tnp = table(repmat(tgt,numel(idxTe),1), repmat(seed,numel(idxTe),1), zeros(numel(idxTe),1), idxTe(:), ...
                mFolderTe, mSubjectTe, mPartTe, mFileTe, ...
                ytrue_idx_np(:), I_np(:), repmat("no_pretraining", numel(idxTe),1), ...
                'VariableNames', {'subject','seed','isTrain','index','folder','subjectStr','part','file','y_true','y_pred','model'});
            allRows = [allRows; Tnp]; %#ok<AGROW>
        end
    else
        fprintf('No target samples available for fine-tuning for subject %s.\n', tgt);
    end

    % Fine-tune entire pretrained model at lower LR
    if nFt > 0
        X_ft_cell = to_cellseq(Xc(:,:,idxFt));
        y_ft_cat  = onehot_to_categorical(y_mat_all(:, idxFt));
        opts_ft = trainingOptions('adam', ...
            'MaxEpochs', 12, ...
            'MiniBatchSize', 32, ...
            'InitialLearnRate', 5e-4, ...
            'Shuffle', 'every-epoch', ...
            'ExecutionEnvironment', 'auto', ...
            'Verbose', false, 'Plots', 'none', ...
            'OutputNetwork', 'last-iteration');

        % Continue training from pretrained weights
        try
            lgraph_ft = layerGraph(net.Layers);
            lgraph_ft = freezeBatchNormalization(lgraph_ft); % helps with tiny batches
            net = trainNetwork(X_ft_cell, y_ft_cat, lgraph_ft, opts_ft);
        catch
            net = trainNetwork(X_ft_cell, y_ft_cat, net.Layers, opts_ft);
        end
    end

    % Evaluate after fine-tune
    if ~isempty(idxTe)
        X_te_cell = to_cellseq(Xc(:,:,idxTe));
        Y_te = y_mat_all(:, idxTe);
        ytrue_idx = onehot_argmax(Y_te);
        ypred_cat = classify(net, X_te_cell);
        I = double(ypred_cat);
        acc_ft = mean(I(:) == ytrue_idx(:));

        % Log detailed test rows
        mFolderTe  = local_meta_col(meta(idxTe), 'folder').';
        mSubjectTe = local_meta_col(meta(idxTe), 'subject').';
        mPartTe    = local_meta_col(meta(idxTe), 'part').';
        mFileTe    = local_meta_col(meta(idxTe), 'file').';
        Tte = table(repmat(tgt,numel(idxTe),1), repmat(seed,numel(idxTe),1), zeros(numel(idxTe),1), idxTe(:), ...
            mFolderTe, mSubjectTe, mPartTe, mFileTe, ...
            ytrue_idx(:), I(:), repmat("fine_tuned", numel(idxTe),1), ...
            'VariableNames', {'subject','seed','isTrain','index','folder','subjectStr','part','file','y_true','y_pred','model'});
        allRows = [allRows; Tte]; %#ok<AGROW>
    else
        acc_ft = NaN;
    end

    % Summary row (now includes acc_ft_only)
    sumRows = [sumRows; {tgt, acc_ft, acc_pre, acc_ftonly, double(nFt)}]; %#ok<AGROW>
    fprintf('Subject %s - nFineTune=%d, Acc FT=%.3f, Acc PRE=%.3f, No-pretrain=%.3f\n', ...
        tgt, nFt, acc_ft, acc_pre, acc_ftonly);
end

% Save CSVs
summaryCsv = fullfile(resultsDir, 'transfer_learning_summary.csv');
detailsCsv = fullfile(resultsDir, 'transfer_learning_predictions.csv');
try
	writetable(sumRows, summaryCsv);
catch ME
	warning(ME.identifier, 'Failed to save TL summary CSV: %s', ME.message);
end
try
	writetable(allRows, detailsCsv);
catch ME
	warning(ME.identifier, 'Failed to save TL details CSV: %s', ME.message);
end

%% ---------------- Local helpers ----------------
function net = pretrain_cnn_in_chunks(X_src, y_src_cat, layers)
% Memory-safe pretraining with multiple passes over all chunks.
    B = size(X_src,3);
    chunkSize = 250;              % reduce if RAM is tight
    mb = 32;                      % small batch
    epochsPerChunk = 3;           % few epochs per chunk
    numPasses = 3;                % pass over all chunks multiple times
    baseLR = 1e-3;                % decay per pass

    net = [];
    for pass = 1:numPasses
        idxAll = randperm(B);     % reshuffle each pass
        lr = baseLR * (0.5)^(pass-1);
        fprintf('  Pretrain pass %d/%d (LR=%.5f)\n', pass, numPasses, lr);

        for s = 1:chunkSize:B
            e = min(B, s+chunkSize-1);
            idx = idxAll(s:e);
            Ni = numel(idx);

            % Build chunk (cast to single)
            Xc = cell(Ni,1);
            for i = 1:Ni
                Xc{i} = single(X_src(:,:,idx(i)));
            end
            yc = y_src_cat(idx);

            opts = trainingOptions('adam', ...
                'MaxEpochs', epochsPerChunk, ...
                'MiniBatchSize', mb, ...
                'InitialLearnRate', lr, ...
                'Shuffle', 'every-epoch', ...
                'ExecutionEnvironment', 'cpu', ...   % switch to 'auto' if GPU RAM allows
                'Verbose', false, 'Plots', 'none', ...
                'OutputNetwork', 'last-iteration');

            if isempty(net) && pass == 1
                net = trainNetwork(Xc, yc, layers, opts);
            else
                net = trainNetwork(Xc, yc, net.Layers, opts);
            end

            clear Xc yc
        end
    end
end

function Xcell = to_cellseq(X)
% Convert [C x T x N] to N-by-1 cell of [C x T] (cast to single).
    N = size(X,3);
    Xcell = cell(N,1);
    for i = 1:N
        Xcell{i} = single(X(:,:,i));
    end
end

function yc = onehot_to_categorical(Y)
% Y: [K x N] one-hot -> Nx1 categorical with categories "1":"K".
    [K, N] = size(Y);
    [~, idx] = max(Y, [], 1);
    yc = categorical(idx(:), 1:K, string(1:K));
end

function idx = onehot_argmax(Y)
% Y: [K x N] -> Nx1 numeric class indices 1..K
    [~, idx] = max(Y, [], 1);
    idx = idx(:);
end

% ---------------- Local meta helpers ----------------
function s = local_meta_col(metaArr, field)
% Return a 1xN string array from metaArr.(field), robust to cells/numerics/empties.
    s = strings(1, numel(metaArr));
    for i = 1:numel(metaArr)
        val = "";
        try
            val = metaArr(i).(field);
        catch
            % leave as ""
        end
        s(i) = local_to_string_scalar(val);
    end
end

function out = local_to_string_scalar(v)
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
                out = local_to_string_scalar(v{1});
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