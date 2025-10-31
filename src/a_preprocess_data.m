%% Exoskeleton preprocessing → data.mat (ALL subjects)
% Configure paths and sampling rates
% Tip: By default, this script uses a repo-relative structure:
%   - Raw data:       data/raw/
%   - Processed data: data/processed/data.mat
% You can override 'main_folder' and 'out_file' in the workspace before running.

if ~exist('main_folder','var') || isempty(main_folder)
    main_folder = fullfile(pwd, 'data', 'raw');
end

fsEMG = 1000;   % Hz
fsIMU = 1000;   % Hz
adc_vref  = 1.1;    % Volts (EMG conversion)
adc_max   = 4095;   % e.g., 12-bit ADC

% Output file (repo-relative by default)
if ~exist('out_file','var') || isempty(out_file)
    out_file = fullfile(pwd, 'data', 'processed', 'data.mat');
end
out_dir  = fileparts(out_file);
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

% Look for CSVs recursively inside each Part? (true recommended)
RECURSIVE_IN_PART = true;

% Initialize output structure (actions as fields)
data = struct('Turn_left', [], 'Turn_right', [], 'Pick_up_object', [], ...
              'Walking_backwards', [], 'Walking_forwards', []);

fprintf('Input root : %s\n', main_folder);
fprintf('Output file: %s\n', out_file);

%% Enumerate top-level action folders
folders = dir(main_folder);
folders = folders([folders.isdir]);
folders = folders(~ismember({folders.name}, {'.','..'}));

% Counters
nActions = 0; nSubjects = 0; nParts = 0; nFiles = 0; nKept = 0;

tic;
for i = 1:numel(folders)
    folder = folders(i).name;
    sub_folder = fullfile(main_folder, folder);
    nActions = nActions + 1;

    % Validate/prepare action field
    dataFieldName = matlab.lang.makeValidName(strrep(folder, ' ', '_'));
    if ~isfield(data, dataFieldName)
        data.(dataFieldName) = [];
    end

    % List subjects
    subjects = dir(sub_folder);
    subjects = subjects([subjects.isdir]);
    subjects = subjects(~ismember({subjects.name}, {'.','..'}));

    for j = 1:numel(subjects)
        subject = subjects(j).name;
        nSubjects = nSubjects + 1;

        subject_folder = fullfile(sub_folder, subject);

        % List parts
        parts = dir(subject_folder);
        parts = parts([parts.isdir]);
        parts = parts(~ismember({parts.name}, {'.','..'}));

        for k = 1:numel(parts)
            part = parts(k).name;
            nParts = nParts + 1;

            part_folder = fullfile(subject_folder, part);

            % Get CSVs in this part (optionally recursive)
            if RECURSIVE_IN_PART
                files = dir(fullfile(part_folder, '**', '*.csv'));  % requires R2016b+
            else
                files = dir(fullfile(part_folder, '*.csv'));
            end

            for l = 1:numel(files)
            % For debugging, you can limit the number of files, e.g.:
            %for l = 1:2
                % Build absolute path to file
                if RECURSIVE_IN_PART
                    data_file = fullfile(files(l).folder, files(l).name);
                else
                    data_file = fullfile(part_folder, files(l).name);
                end

                nFiles = nFiles + 1;

                try
                    df = readtable(data_file);
                catch ME
                    warning("Skipping file (read error): %s\n%s", data_file, ME.message);
                    continue
                end

                % Process only numeric, non-time columns
                vars = df.Properties.VariableNames;
                for m = 1:numel(vars)
                    vname = vars{m};
                    col   = df{:, m};

                    % Skip non-numeric or obvious time/id columns
                    if ~isnumeric(col) || any(strcmpi(vname, ["time","timestamp","idx","id"]))
                        continue
                    end

                    if contains(vname, 'EMG', 'IgnoreCase', true)
                        df{:, m} = EMG_data_processing(col, fsEMG, adc_vref, adc_max);
                    elseif contains(vname, 'IMU', 'IgnoreCase', true)
                        df{:, m} = IMU_data_processing(col, fsIMU);
                    else
                        % Unknown channel → leave as-is (or choose a default)
                    end
                end

                % Build record with metadata
                rec = struct( ...
                    'table',   df, ...
                    'subject', string(subject), ...
                    'part',    string(part), ...
                    'file',    string(files(l).name), ...
                    'folder',  string(folder) ...
                );

                % Append to the corresponding action field
                data.(dataFieldName) = [data.(dataFieldName), rec]; %#ok<AGROW>
                nKept = nKept + 1;

                % Progress print
                fprintf('[%s | %s | %s] %s\n', folder, subject, part, files(l).name);
            end
        end
    end
end
tElapsed = toc;

%% Save (v7.3 for large structs)
save(out_file, 'data', '-v7.3');

% Summary
fprintf('\n=== Preprocessing complete ===\n');
fprintf('Actions:   %d\n', nActions);
fprintf('Subjects:  %d\n', nSubjects);
fprintf('Parts:     %d\n', nParts);
fprintf('Files seen:%d | Files kept:%d\n', nFiles, nKept);
fprintf('Elapsed:   %.1f s\n', tElapsed);

% Per-action breakdown
actNames = fieldnames(data);
for a = 1:numel(actNames)
    nrec = numel(data.(actNames{a}));
    fprintf('  %-20s : %d traces\n', actNames{a}, nrec);
end

% Print the structure of the data variable
fprintf('\n=== Data Structure ===\n');
disp(data);

%% ------------- Helper functions -------------
function [sos, g] = butter_bandpass(lowcut, highcut, fs, order)
    nyq = 0.5*fs;
    Wn  = sort([lowcut, highcut]/nyq);
    Wn(Wn<=0) = eps; Wn(Wn>=1) = 1 - eps;
    [z,p,k] = butter(order, Wn, 'bandpass');
    [sos,g] = zp2sos(z,p,k);
end

function EMG = EMG_data_processing(x, fs, vref, adc_max)
    if nargin < 3, vref = 1.1; end
    if nargin < 4, adc_max = 4095; end
    v = double(x) * vref / adc_max;            % ADC → volts (parametric)

    % Robust outlier suppression (~50 samples @ fs=1000)
    [v_hamp, ~] = hampel(v, 50, 3.0);

    % Bandpass 0.2–400 Hz
    [sos,g] = butter_bandpass(0.2, 400, fs, 5);
    filtered = filtfilt(sos, g, v_hamp);

    mu = mean(filtered, 'omitnan');
    sg = std(filtered, 0, 'omitnan'); if sg==0 || isnan(sg), sg = 1; end
    EMG = (filtered - mu) / sg;
end

function IMU = IMU_data_processing(x, fs)
    [sos,g] = butter_bandpass(0.2, 10, fs, 5); % Bandpass 0.2–10 Hz
    filtered = filtfilt(sos, g, double(x));

    mu = mean(filtered, 'omitnan');
    sg = std(filtered, 0, 'omitnan'); if sg==0 || isnan(sg), sg = 1; end
    IMU = (filtered - mu) / sg;
end
