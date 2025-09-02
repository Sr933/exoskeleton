%% ================================================================
% CNN sliding-window time-series classification (mode = "all")
% - Train CNN as usual (5 seeds)
% - Concatenate test samples into one long [C x T_total] signal
% - Sliding-window predictions with stride = 1000 and 2000 (window = CNN input width)
% - Accuracy as "percentage of time" (fraction of window centers correct)
% - Kalman smoothing of per-class scores to compare smoothed vs raw accuracy
% - Saves results to results/timeseries_cnn_time_accuracy_shiftXXXX.csv
% ================================================================
clear; clc;

% ---------------- Paths ----------------
scriptFullPath = mfilename('fullpath');
scriptDir = fileparts(scriptFullPath);
projectRoot = fileparts(scriptDir);
matFile    = fullfile(projectRoot, 'Preprocessed data', 'data.mat');
resultsDir = fullfile(projectRoot, 'results');
if ~exist(resultsDir, 'dir'); mkdir(resultsDir); end

% ---------------- Config ----------------
channelMode = "all";             % fixed as requested
seeds       = [1 2 3 4 5];
strides     = [1000 2000];       % compare strides
kalmanQ     = 2e-3;              % process noise variance for scores
kalmanR     = 2e-2;              % measurement noise variance for scores

% ---------------- Load data ----------------
S = load(matFile);
assert(isfield(S,'data'), 'Expected variable ''data'' in MAT file.');
data = S.data;

label_map = exoskeleton_library.label_mapping();
[Xc, y, meta] = exoskeleton_library.load_data_CNN(data, label_map, channelMode);
assert(~isempty(Xc) && ~isempty(y), 'No valid data.');

% ---------------- Loop seeds ----------------
for stride = strides
    fprintf('\n=== Sliding-window stride = %d ===\n', stride);

    perSeed = table('Size',[numel(seeds) 4], ...
        'VariableTypes', {'double','double','double','double'}, ...
        'VariableNames', {'seed','accuracy_raw_pct','accuracy_kalman_pct','n_windows'});

    for si = 1:numel(seeds)
        seed = seeds(si);
        rng(seed);

        % Split into train/test (same as normal CNN run)
        [Xtr, Xte, ytr, yte, idxTr, idxTe] = exoskeleton_library.split_data_CNN(Xc, y, seed);
        assert(~isempty(Xtr) && ~isempty(Xte), 'Empty split for seed %d.', seed);

        % Convert one-hot cells -> categorical (required by trainNetwork)
        if iscell(ytr)
            [ytr, K] = onehot_cells_to_categorical(ytr);
            yte      = onehot_cells_to_categorical(yte, K);
        end

        % Model
        C = size(Xtr{1},1);
        W = size(Xtr{1},2);             % input window length (time)
        layers  = exoskeleton_library.CNN(C);
        options = exoskeleton_library.CNN_training_options();

        % Train
        trainedNet = trainNetwork(Xtr, ytr, layers, options);

        % Map yte (categorical) to numeric class indices consistent with the network output
        if isprop(trainedNet.Layers(end), 'Classes')
            clsNames = string(trainedNet.Layers(end).Classes);
        else
            clsNames = string(categories(ytr));
        end
        yte_names = string(yte);
        [~, yte_idx] = ismember(yte_names, clsNames);  % 1..K per segment

        % -------- Memory-safe streaming inference (no X_long) --------
        % Segment lengths and cumulative ends
        L = zeros(numel(Xte),1);
        for ii = 1:numel(Xte), L(ii) = size(Xte{ii},2); end
        edges = cumsum(L);
        Ttot  = edges(end);
        if Ttot < W
            warning('Seed %d: total test length (%d) < window (%d). Skipping.', seed, Ttot, W);
            perSeed{si,:} = [seed, NaN, NaN, 0];
            continue;
        end
        halfW = floor(W/2);
        startC = 1 + halfW;
        endC   = Ttot - (W - halfW);
        if endC < startC
            warning('Seed %d: no valid centers.', seed);
            perSeed{si,:} = [seed, NaN, NaN, 0];
            continue;
        end
        nWin = floor((endC - startC)/stride) + 1;

        % Streaming prediction with small batches
        batchSize = 64;
        correct_raw = 0; correct_k = 0; seen = 0;
        % Kalman filter running state (per class)
        kState.x = []; kState.P = [];

        % Helper for advancing segment pointer
        segIdx = find(edges >= startC, 1, 'first');
        if isempty(segIdx), segIdx = 1; end

        for base = 0:batchSize:(nWin-1)
            nb = min(batchSize, nWin - base);
            Xbatch = cell(nb,1);
            ytrueC = zeros(nb,1);
            for bi = 1:nb
                c = startC + (base+bi-1)*stride;          % center (absolute)
                % Segment containing center (for ground truth)
                segC = find(edges >= c, 1, 'first');
                ytrueC(bi) = yte_idx(segC);               % numeric label at center (1..K)

                % Extract window [C x W] around center spanning segments if needed
                s = c - halfW; e = s + W - 1;             % absolute window bounds
                cur = s;
                j = find(edges >= s, 1, 'first');         % segment containing 's'
                if isempty(j), j = numel(edges); end
                Xw = zeros(C, W, 'like', Xtr{1});
                fillPos = 1;
                while cur <= e
                    % Ensure j points to segment containing 'cur'
                    while j < numel(edges) && cur > edges(j), j = j + 1; end
                    leftBound = leftBoundOfSegment(j, edges);    % 1-based start of segment j

                    offset = cur - leftBound + 1;                % start index in segment j
                    % Guard offsets to valid range
                    if offset < 1
                        offset = 1;
                    elseif offset > L(j)
                        j = min(j+1, numel(edges));
                        continue;
                    end
                    take = min(e - cur + 1, L(j) - offset + 1);  % samples from this segment
                    if take <= 0
                        j = min(j+1, numel(edges));
                        continue;
                    end
                    Xseg = Xte{j};
                    Xw(:, fillPos:fillPos+take-1) = Xseg(:, offset:offset+take-1);
                    fillPos = fillPos + take;
                    cur = cur + take;
                    if cur > e, break; end
                    j = j + 1;                              % next segment
                end
                Xbatch{bi} = Xw;
            end

            % Predict scores for batch with small MiniBatchSize
            S = predict(trainedNet, Xbatch, 'MiniBatchSize', 32);
            if iscell(S), S = cell2mat(S); end
            % Ensure S is [K x nb]
            if size(S,1) == nb, S = S.'; end
            if isempty(kState.x)
                Kcls = size(S,1);
                kState.x = S(:,1);           % init from first window
                kState.P = kalmanR * ones(Kcls,1,'like',S);
            end
            % Update accuracies (raw and Kalman) sequentially
            for bi = 1:nb
                sCol = S(:,bi);
                % raw
                [~, pr] = max(sCol);
                correct_raw = correct_raw + double(pr == double(ytrueC(bi)));
                % Kalman update per class (online)
                [kState, sSm] = kalman_update_online(kState, sCol, kalmanQ, kalmanR);
                [~, pk] = max(sSm);
                correct_k = correct_k + double(pk == double(ytrueC(bi)));
            end
            seen = seen + nb;
        end

        % Accuracy as percent of sampled time (centers)
        acc_raw = (correct_raw / seen) * 100;
        acc_k   = (correct_k   / seen) * 100;

        fprintf('Seed %d: W=%d, T=%d, nWin=%d -> Raw=%.2f%%, Kalman=%.2f%%\n', ...
            seed, W, Ttot, nWin, acc_raw, acc_k);

        perSeed{si,:} = [seed, acc_raw, acc_k, double(nWin)];
    end

    % Save per-stride summary include the channel selection
    outCsv = fullfile(resultsDir, sprintf('timeseries_cnn_accuracy_%s_stride%d.csv', channelMode, stride));
    try
        writetable(perSeed, outCsv);
        fprintf('Saved: %s\n', outCsv);
        % Print mean±std
        mr = mean(perSeed.accuracy_raw_pct,'omitnan');
        sr = std(perSeed.accuracy_raw_pct,'omitnan');
        mk = mean(perSeed.accuracy_kalman_pct,'omitnan');
        sk = std(perSeed.accuracy_kalman_pct,'omitnan');
        fprintf('Stride %d: Raw %.2f±%.2f%% | Kalman %.2f±%.2f%% (n=%d seeds)\n', ...
            stride, mr, sr, mk, sk, sum(isfinite(perSeed.accuracy_raw_pct)));
    catch ME
        warning('Failed to save %s: %s', outCsv, ME.message);
    end
end

%% ---------------- Helpers ----------------
function [Xlong, ytl] = build_long_timeline(Xcells, ycats)
% Concatenate cell windows along time and build ground-truth timeline at sample resolution.
    n = numel(Xcells);
    C = size(Xcells{1},1);
    W = size(Xcells{1},2);
    Ttot = n * W;
    Xlong = zeros(C, Ttot, 'like', Xcells{1});
    ytl   = zeros(Ttot, 1);
    pos = 1;
    for i = 1:n
        Xi = Xcells{i};
        li = size(Xi,2);
        Xlong(:, pos:pos+li-1) = Xi;
        yi = double(ycats(i));          % numeric class index per segment
        ytl(pos:pos+li-1) = yi;
        pos = pos + li;
    end
end

function S_smooth = kalman_smooth_scores(S, qVar, rVar)
% 1D Kalman filter per class over time on score stream.
% Input S: [K x T] raw scores (e.g., softmax probabilities)
% Output:  [K x T] smoothed scores
    [K, T] = size(S);
    S_smooth = zeros(K, T, 'like', S);
    for k = 1:K
        x = S(k,1);           % state (score)
        P = rVar;             % initial variance
        for t = 1:T
            z = S(k,t);       % measurement
            % Predict
            x_pred = x;
            P = P + qVar;
            % Update
            Kgain = P / (P + rVar);
            x = x_pred + Kgain * (z - x_pred);
            P = (1 - Kgain) * P;
            S_smooth(k,t) = x;
        end
    end
    % Optional: renormalize columns to sum to 1 (scores behave like probs)
    colSums = sum(S_smooth, 1);
    colSums(colSums == 0) = 1;
    S_smooth = S_smooth ./ colSums;
end

function [state, sSmooth] = kalman_update_online(state, z, qVar, rVar)
% One-step Kalman update for a K-dim score vector z (probabilities/logits).
% state.x, state.P are Kx1 vectors; returns updated state and smoothed sSmooth.
    x = state.x; P = state.P;
    % Predict
    P = P + qVar;
    % Update (elementwise 1D filters)
    Kg = P ./ (P + rVar);
    x = x + Kg .* (z - x);
    P = (1 - Kg) .* P;
    % Normalize to sum 1
    s = x ./ max(sum(x), eps('like',x));
    state.x = x; state.P = P; sSmooth = s;
end

% Add at end of file with other helpers
function lb = leftBoundOfSegment(j, edges)
% Return 1-based left bound (global) of segment j without invalid indexing
    if j <= 1
        lb = 1;
    else
        lb = edges(j-1) + 1;
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