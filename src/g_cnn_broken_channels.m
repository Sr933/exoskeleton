%% ================================================================
% CNN with simulated sensor failures (test-time only).
% For each seed:
%   - Train on original unmasked data
%   - For each scenario, mask TEST data only and evaluate
% Writes per-scenario details CSVs and summaries.
% ================================================================
clear; clc;

% ---------------- Paths ----------------
scriptFullPath = mfilename('fullpath');
scriptDir = fileparts(scriptFullPath);
projectRoot = fileparts(scriptDir); % parent folder
matFile = fullfile(projectRoot, 'Preprocessed data', 'data.mat');
resultsDir = fullfile(projectRoot, 'results');
if ~exist(resultsDir, 'dir'); mkdir(resultsDir); end

% ---------------- Config ----------------
scenarios = ["emg_all","emg_1_4","emg_5_8","imu_1","imu_2","imu_3","imu_1_3"]; % add "none" to include baseline
channelMode = "all";   % must be "all" to zero specific groups

% ---------------- Load data (as in CNN) ----------------
S = load(matFile);
assert(isfield(S,'data'), 'Expected variable ''data'' in MAT file.');
data = S.data;
label_map = exoskeleton_library.label_mapping();
[Xc, y, meta] = exoskeleton_library.load_data_CNN(data, label_map, channelMode);
assert(~isempty(Xc) && ~isempty(y), 'No valid data.');

% Build per-sample channel masks aligned with Xc rows
chanMasks = build_channel_masks(data);
assert(numel(chanMasks) == size(Xc,3), 'Channel masks do not align with Xc pages.');

% ---------------- Prepare per-scenario outputs ----------------
scList = string(scenarios);
nSc = numel(scList);
detailsCsv = strings(nSc,1);
wroteHeader = false(nSc,1);
for j = 1:nSc
    detailsCsv(j) = fullfile(resultsDir, sprintf('cnn_broken_%s_predictions_all_seeds.csv', scList(j)));
    % overwrite any existing file on first write
    if exist(detailsCsv(j), 'file'), delete(detailsCsv(j)); end
end
seeds = [1 2 3 4 5];
accMat = nan(numel(seeds), nSc);

% ---------------- Loop seeds: train once, test many scenarios ----------------
for si = 1:numel(seeds)
    seed = seeds(si);
    fprintf('\n=== Seed %d (train on original data) ===\n', seed);

    % Split (unmasked)
    [Xtr, Xte, ytr, yte, idxTr, idxTe] = exoskeleton_library.split_data_CNN(Xc, y, seed);

    % Convert one-hot cells -> categorical (ensure 5 classes)
    if iscell(ytr)
        [ytr, K] = onehot_cells_to_categorical(ytr);   % K should be 5
        yte      = onehot_cells_to_categorical(yte, K);
    end

    % Model & options
    inpSize = size(Xtr{1},1);
    layers  = exoskeleton_library.CNN(inpSize);
    options = exoskeleton_library.CNN_training_options(); % headless

    % Train on original (unmasked) training data
    trainedNet = trainNetwork(Xtr, ytr, layers, options);

    % Evaluate each scenario by masking TEST ONLY
    for j = 1:nSc
        sc = scList(j);
        % Mask test set only
        Xte_mask = apply_mask_to_cells(Xte, chanMasks(idxTe), sc);

        % Quick preview of channels masked (on this test subset)
        maskedPerSample = arrayfun(@(k) nnz(get_mask_for_mode(chanMasks{idxTe(k)}, sc)), 1:numel(idxTe))';
        fprintf('  [%s] masked/test sample: mean=%.2f (C=%d)\n', sc, mean(maskedPerSample), inpSize);

        % Predict and accuracy
        YPred = classify(trainedNet, Xte_mask);
        accuracy = sum(YPred(:) == yte(:)) / numel(yte);
        accMat(si, j) = accuracy;
        fprintf('  [%s] Acc: %.2f%%\n', sc, accuracy*100);

        % Build test-only rows
        yte_idx        = double(yte);
        ypred_te_idx   = double(YPred);
        Tte = table(repmat(sc,numel(idxTe),1), repmat(seed,numel(idxTe),1), zeros(numel(idxTe),1), idxTe(:), ...
            [meta(idxTe).folder].', [meta(idxTe).subject].', [meta(idxTe).part].', [meta(idxTe).file].', ...
            yte_idx(:), ypred_te_idx(:), ...
            'VariableNames', {'scenario','seed','isTrain','index','folder','subject','part','file','y_true','y_pred'});

        % Stream to per-scenario CSV
        if ~wroteHeader(j)
            writetable(Tte, detailsCsv(j));               % write header
            wroteHeader(j) = true;
        else
            try
                writetable(Tte, detailsCsv(j), 'WriteMode','append', 'WriteVariableNames', false);
            catch
                % Fallback for older MATLAB
                existing = readtable(detailsCsv(j));
                existing = [existing; Tte]; %#ok<AGROW>
                writetable(existing, detailsCsv(j));
                clear existing;
            end
        end

        % Free per-scenario temporaries
        clear Xte_mask YPred Tte yte_idx ypred_te_idx maskedPerSample
    end

    % Free per-seed memory (and GPU cache if used)
    clear Xtr Xte trainedNet
    try, reset(gpuDevice); catch, end
end

% ---------------- Save per-scenario summaries ----------------
for j = 1:nSc
    Ts = table(seeds(:), accMat(:,j), 'VariableNames', {'seed','accuracy'});
    summaryCsv = fullfile(resultsDir, sprintf('cnn_broken_%s_accuracy_summary.csv', scList(j)));
    try, writetable(Ts, summaryCsv); catch ME, warning('Save summary failed (%s): %s', scList(j), ME.message); end
    fprintf('Saved: %s, %s\n', summaryCsv, detailsCsv(j));
end

%% ---------------- Helpers ----------------
function idx = get_mask_for_mode(m, mode)
    switch string(mode)
        case "emg_all", idx = m.emg_all;
        case "emg_1_4", idx = m.emg_1_4;
        case "emg_5_8", idx = m.emg_5_8;
        case "imu_1",   idx = m.imu_1;
        case "imu_2",   idx = m.imu_2;
        case "imu_3",   idx = m.imu_3;
        case "imu_1_3", idx = m.imu_1_3;
        case "none",    idx = false(size(m.emg_all));
        otherwise, error('Unknown failure mode: %s', mode);
    end
end

function XcMasked = apply_mask_to_cells(XcCells, masksSubset, mode)
% Zero rows in cell windows according to per-sample masks (no 3D copy).
    XcMasked = XcCells;
    for i = 1:numel(XcCells)
        m = get_mask_for_mode(masksSubset{i}, mode);
        if any(m)
            Xi = XcCells{i};
            Xi(m, :) = 0;
            XcMasked{i} = Xi;
        end
    end
end

function masks = build_channel_masks(data)
% Build per-sample masks for channel groups following the same iteration as load_data_CNN (mode=all).
    keys = fieldnames(data);
    masks = {};
    for ki = 1:numel(keys)
        k = keys{ki};
        recs = data.(k);
        if ~isstruct(recs), continue; end
        for ri = 1:numel(recs)
            r = recs(ri);
            if isfield(r,'table') && istable(r.table)
                T = r.table;
            elseif isfield(r,'tbl') && istable(r.tbl)
                T = r.tbl;
            elseif istable(r)
                T = r;
            else
                continue
            end
            varNames = string(T.Properties.VariableNames);
            vnLower = lower(varNames);
            isTime = contains(vnLower,'time') | contains(vnLower,'stamp');
            usedNames = varNames(~isTime);  % order should match Xc rows
            nC = numel(usedNames);

            % Strict EMG detection: only EMG1..EMG8
            emgID = nan(nC,1);
            for ii = 1:nC
                nm = usedNames(ii);
                tok = regexp(nm, '^(?i)emg[\s_-]*0?([1-8])$', 'tokens','once');
                if ~isempty(tok), emgID(ii) = str2double(tok{1}); end
            end
            emg_all = isfinite(emgID);
            emg_1_4 = emg_all & emgID >= 1 & emgID <= 4;
            emg_5_8 = emg_all & emgID >= 5 & emgID <= 8;

            % IMU grouping
            imuID = nan(nC,1);
            for ii = 1:nC
                if emg_all(ii), continue; end
                nm = string(usedNames(ii));
                tok = regexp(nm, '^[A-Za-z]{2}[\s_-]*0?([1-3])$', 'tokens','once');
                if ~isempty(tok)
                    imuID(ii) = str2double(tok{1});
                else
                    tok2 = regexp(nm, '(?<!\d)([1-3])$', 'tokens','once');
                    if ~isempty(tok2), imuID(ii) = str2double(tok2{1}); end
                end
            end
            imu_1 = ~emg_all & (imuID == 1);
            imu_2 = ~emg_all & (imuID == 2);
            imu_3 = ~emg_all & (imuID == 3);

            m.emg_all = emg_all;
            m.emg_1_4 = emg_1_4;
            m.emg_5_8 = emg_5_8;
            m.imu_1   = imu_1;
            m.imu_2   = imu_2;
            m.imu_3   = imu_3;
            m.imu_1_3 = imu_1 | imu_3;

            masks{end+1} = m; %#ok<AGROW>
        end
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