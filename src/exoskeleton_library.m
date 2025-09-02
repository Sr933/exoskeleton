classdef (Abstract) exoskeleton_library
% Static utilities for loading data, splitting, and building models

    methods (Static)
        function m = label_mapping()
        % 5-class mapping used across the project
            m = containers.Map( ...
                {'Turn_left','Turn_right','Pick_up_object','Walking_backwards','Walking_forwards'}, ...
                num2cell(1:5));
        end

        function layers = CNN(InputSize)
        % Initial CNN architecture (sequence-to-label), 5 classes
            layers = [ ...
                sequenceInputLayer(InputSize, MinLength=1)

                convolution1dLayer(9, 10, Padding="same")
                batchNormalizationLayer
                reluLayer
                maxPooling1dLayer(50, Stride=50, Padding="same")

                convolution1dLayer(9, 20, Padding="same")
                batchNormalizationLayer
                reluLayer
                maxPooling1dLayer(10, Stride=10, Padding="same")

                convolution1dLayer(9, 30, Padding="same")
                batchNormalizationLayer
                reluLayer
                maxPooling1dLayer(10, Stride=10, Padding="same")

                convolution1dLayer(9, 40, Padding="same")
                batchNormalizationLayer
                reluLayer

                % Collapse time dimension -> sequence-to-label
                globalAveragePooling1dLayer

                dropoutLayer(0.5)
                fullyConnectedLayer(5)
                softmaxLayer
                classificationLayer
            ];
        end

        function options = CNN_training_options()
        % Options compatible with trainNetwork
            options = trainingOptions("adam", ...
                InitialLearnRate=0.0015, ...
                MaxEpochs=15, ...
                Shuffle="every-epoch", ...
                MiniBatchSize=50, ...
                ExecutionEnvironment="auto", ...
                Plots="none", ...
                Verbose=false);
        end

        % ---------------- LSTM baseline ----------------
        function layers = LSTM(InputSize)
        % Simple LSTM sequence classifier (sequence-to-label), 5 classes
            layers = [ ...
                sequenceInputLayer(InputSize, MinLength=1)
                % Downsample in time by 50 via conv + pooling
                convolution1dLayer(9, 20, Padding="same")
                batchNormalizationLayer
                reluLayer
                maxPooling1dLayer(100, Stride=100, Padding="same")
                % Recurrent head
                lstmLayer(100, OutputMode="last")
                dropoutLayer(0.5)
                fullyConnectedLayer(5)
                softmaxLayer
                classificationLayer ];
        end

        function options = LSTM_training_options()
        % LSTM training options
            options = trainingOptions("adam", ...
                InitialLearnRate=0.0015, ...
                MaxEpochs=15, ...
                Shuffle="every-epoch", ...
                MiniBatchSize=64, ...
                GradientThreshold=1, ...
                ExecutionEnvironment="auto", ...
                Plots="none", ...
                Verbose=false);
        end

        function [Xtr, Xte, ytr, yte, idxTr, idxTe] = split_data_CNN(Xc, y, seed)
            if nargin < 3 || isempty(seed), seed = 1; end
            rng(seed);

            % Normalize inputs to 1xN cell of [C x T]
            if isnumeric(Xc) && ndims(Xc) == 3
                N = size(Xc,3);
                Xcells = cell(1,N);
                for i=1:N, Xcells{i} = Xc(:,:,i); end
            elseif iscell(Xc)
                Xcells = Xc;
            else
                error('Xc must be 1xN cell of [C x T] or [C x T x N] numeric array.');
            end

            N = numel(Xcells);
            cv = cvpartition(N, 'HoldOut', 0.5);
            idxTr = find(training(cv));
            idxTe = find(test(cv));

            Xtr = Xcells(idxTr);
            Xte = Xcells(idxTe);

            % Ensure responses are categorical labels (sequence-to-label)
            if iscell(y)
                % Convert one-hot numeric rows to class indices
                if all(cellfun(@isnumeric, y))
                    idxAll = zeros(numel(y),1);
                    K = 0;
                    for ii = 1:numel(y)
                        r = y{ii};
                        r = r(:).';                % row
                        [~, k] = max(r);
                        if isempty(k) || all(r==0), k = 1; end
                        idxAll(ii) = k;
                        K = max(K, numel(r));
                    end
                    cats = string(1:max(K, max(idxAll)));
                    yAll = categorical(idxAll, 1:numel(cats), cats);
                    ytr = yAll(idxTr);
                    yte = yAll(idxTe);
                elseif all(cellfun(@iscategorical, y))
                    % Already categorical per-trial; reduce to single label if needed
                    yAll = vertcat(y{:});
                    ytr = yAll(idxTr);
                    yte = yAll(idxTe);
                else
                    error('Unsupported response cell format.');
                end
            else
                % Numeric vector -> categorical; categorical passed through
                if isnumeric(y)
                    K = max(y(:));
                    cats = string(1:K);
                    y = categorical(y(:), 1:K, cats);
                end
                ytr = y(idxTr);
                yte = y(idxTe);
            end
        end

        function [Xcells, y, meta] = load_data_CNN(data, label_mapping, channelMode)
        % Load struct of class->trials (tables) to:
        %   Xcells: 1xN cell of [C x T] numeric
        %   y:      1xN cell of one-hot row vectors (1xK)
        %   meta:   struct array with file info (best-effort)
        % channelMode: "all" | "emg" | "imu" | "right_leg" | "left_leg"
            if nargin < 3 || strlength(string(channelMode)) == 0
                channelMode = "all";
            else
                channelMode = string(channelMode);
            end
            K = exoskeleton_library.local_num_classes(label_mapping);

            Xcells = {};
            y = {};
            meta = struct('folder', strings(0,1), 'subject', strings(0,1), ...
                          'part',   strings(0,1), 'file',    strings(0,1));

            keys = fieldnames(data);
            refNames = string.empty(0,1); % enforce consistent channel order across trials

            for ki = 1:numel(keys)
                labelKey = string(keys{ki});
                trials = data.(keys{ki});
                if isempty(trials), continue; end

                % Iterate trials across cells, struct arrays, or single table
                if iscell(trials)
                    nTrials = numel(trials);
                    getTrial = @(j) trials{j};
                elseif istable(trials)
                    nTrials = 1;
                    getTrial = @(j) trials; %#ok<NASGU>
                elseif isstruct(trials)
                    nTrials = numel(trials);
                    getTrial = @(j) trials(j);
                else
                    continue
                end

                for tj = 1:nTrials
                    tObj = getTrial(tj);
                    T = exoskeleton_library.local_extract_table(tObj);
                    if ~istable(T), continue; end

                    % Candidate channel names (exclude time-like)
                    varNames = string(T.Properties.VariableNames);
                    vnLower = lower(varNames);
                    isTime = contains(vnLower,'time') | contains(vnLower,'stamp') | contains(vnLower,'sample');
                    cand = varNames(~isTime);

                    % Select variables by channelMode; fallback to all non-time
                    used = exoskeleton_library.local_select_by_mode(cand, channelMode);
                    if isempty(used), used = cand; end

                    % Lock reference order from the first valid trial
                    if isempty(refNames)
                        refNames = used(:);
                    end

                    % Ensure current trial has all required channels (same order)
                    [tf, ~] = ismember(refNames, used);
                    if ~all(tf)
                        % Skip trial if missing required channels
                        continue
                    end

                    % Use the reference ordering
                    Tuse = T(:, refNames);
                    Xi = table2array(Tuse).';  % [C x T]
                    Xcells{end+1} = Xi; %#ok<AGROW>

                    % One-hot label
                    kIdx = exoskeleton_library.local_map_label_to_index(label_mapping, labelKey);
                    oh = zeros(1,K); if kIdx>=1 && kIdx<=K, oh(kIdx) = 1; end
                    y{end+1} = oh; %#ok<AGROW>

                    % Meta (best effort)
                    meta(end+1).folder  = ""; %#ok<AGROW>
                    meta(end).subject   = "";
                    meta(end).part      = "";
                    meta(end).file      = "";
                end
            end

            % Row vector cell for y
            y = y(:).';
        end
    end

    methods (Static, Access=private)
        function used = local_select_by_mode(cand, channelMode)
        % Channel selection:
        % - EMG if name contains "emg" (case-insensitive), else IMU
        % - Sensor index = trailing digits at end of the variable name (e.g., "...7" -> 7)
        % - Right leg:  IMU 1 & 3 + EMG 1–4
        % - Left  leg:  IMU 2     + EMG 5–8
            names = string(cand(:));
            lowerNames = lower(names);
            isEmg = contains(lowerNames, "emg");
            isImu = ~isEmg;

            % Extract trailing numeric index from each name
            numIdx = nan(numel(names),1);
            for i = 1:numel(names)
                tok = regexp(names(i), '(\d+)$', 'tokens', 'once');
                if ~isempty(tok)
                    numIdx(i) = str2double(tok{1});
                end
            end

            mode = lower(string(channelMode));
            switch mode
                case "emg"
                    sel = find(isEmg);
                    [~, ord] = sort(numIdx(sel));
                    used = names(sel(ord));

                case "imu"
                    sel = find(isImu);
                    [~, ord] = sort(numIdx(sel));
                    used = names(sel(ord));

                case "right_leg"
                    keepIMU = isImu & ismember(numIdx, [1 3]);
                    keepEMG = isEmg & ismember(numIdx, 1:4);

                    imuIdx = find(keepIMU); [~, iord] = sort(numIdx(imuIdx)); imuList = names(imuIdx(iord));
                    emgIdx = find(keepEMG); [~, eord] = sort(numIdx(emgIdx)); emgList = names(emgIdx(eord));
                    used = [imuList; emgList];

                case "left_leg"
                    keepIMU = isImu & (numIdx == 2);
                    keepEMG = isEmg & ismember(numIdx, 5:8);

                    imuIdx = find(keepIMU); [~, iord] = sort(numIdx(imuIdx)); imuList = names(imuIdx(iord));
                    emgIdx = find(keepEMG); [~, eord] = sort(numIdx(emgIdx)); emgList = names(emgIdx(eord));
                    used = [imuList; emgList];

                otherwise
                    used = names;
            end

            used = used(:);
        end

        function K = local_num_classes(mapping)
            if isa(mapping, 'containers.Map')
                vals = cell2mat(mapping.values);
                K = max(vals);
            elseif iscell(mapping) || isstring(mapping)
                K = numel(mapping);
            else
                K = 5;
            end
        end

        function idx = local_map_label_to_index(mapping, key)
            if isa(mapping, 'containers.Map') && isKey(mapping, char(key))
                idx = mapping(key);
            else
                % Try numeric tail in key, else 0
                tok = regexp(string(key), '(\d+)$', 'tokens', 'once');
                if ~isempty(tok)
                    idx = str2double(tok{1});
                else
                    idx = 0;
                end
            end
        end

        function [isEmg, numIdx] = local_detect_emg(varNames)
        % Detect EMG channels EMG1..EMG8 from names
            isEmg = false(numel(varNames),1);
            numIdx = nan(numel(varNames),1);
            for ii = 1:numel(varNames)
                v = varNames(ii);
                tok = regexp(v, '(?i)\bemg[\s_-]*0?([1-8])\b', 'tokens', 'once');
                if ~isempty(tok)
                    isEmg(ii) = true;
                    numIdx(ii) = str2double(tok{1});
                end
            end
        end

        function [isImu, numIdx] = local_detect_imu(varNames)
        % Detect IMU sensor index IMU1, IMU2, IMU3, ...
            isImu = false(numel(varNames),1);
            numIdx = nan(numel(varNames),1);
            for ii = 1:numel(varNames)
                v = varNames(ii);
                tok = regexp(v, '(?i)\bimu[\s_-]*0?([0-9]+)\b', 'tokens', 'once');
                if ~isempty(tok)
                    isImu(ii) = true;
                    numIdx(ii) = str2double(tok{1});
                end
            end
        end

        function T = local_extract_table(tObj)
        % Extract a MATLAB table from many possible trial container shapes
            T = [];
            if istable(tObj)
                T = tObj; return;
            end
            if isstruct(tObj)
                fns = fieldnames(tObj);
                % Prefer common names if present
                pref = ["table","tbl","T","data","Data"];
                for p = pref
                    if isfield(tObj, p) && istable(tObj.(p))
                        T = tObj.(p); return;
                    end
                end
                % Else first table-valued field
                for k = 1:numel(fns)
                    v = tObj.(fns{k});
                    if istable(v)
                        T = v; return;
                    end
                end
            end
        end

        

        function yIdx = local_to_class_indices(y)
        % Convert responses to numeric class indices (1..K)
            if iscell(y)
                if all(cellfun(@isnumeric, y))
                    K = 0; idx = zeros(numel(y),1);
                    for ii=1:numel(y)
                        r = y{ii}(:).'; [~, k] = max(r);
                        if isempty(k) || all(r==0), k = 1; end
                        idx(ii) = k; K = max(K, numel(r));
                    end
                    yIdx = idx;
                elseif all(cellfun(@iscategorical, y))
                    yy = vertcat(y{:});
                    [~,~,yIdx] = unique(yy);
                else
                    error('Unsupported response format in cell array.');
                end
            elseif iscategorical(y)
                [~,~,yIdx] = unique(y);
            elseif isnumeric(y)
                yIdx = y(:);
            else
                error('Unsupported response type for labels.');
            end
        end
    end
end



