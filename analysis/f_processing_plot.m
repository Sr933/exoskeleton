% Publication-quality processing panels (saved individually) using the same filters as a_preprocess_data.m
clear; clc;

% ---------------- Paths ----------------
scriptFullPath = mfilename('fullpath');
scriptDir = fileparts(scriptFullPath);
projectRoot = fileparts(scriptDir);    % parent of "analysis"
plotsDir   = fullfile(projectRoot, 'plots');
if ~exist(plotsDir, 'dir'); mkdir(plotsDir); end

% ---------------- Data -----------------
filename = "C:\Users\silas\Downloads\Exoskeleton Trial Data Set\Walking forwards\Combined_trial92_merged.csv";
T = readtable(filename);

% Detect sampling rate (fallback to 1000 Hz for time axis and filters)
[~, fs] = detect_time_and_fs(T, 1000);

% Select channels (adjust as needed)
imuCol = 3;   % IMU channel to visualize
emgCol = 35;  % EMG channel to visualize

imu_raw = T{:, imuCol};
emg_adc = T{:, emgCol};

% Time axes in seconds (divide by 1000 if fs=1000)
tIMU = (0:numel(imu_raw)-1)'/fs;
tEMG = (0:numel(emg_adc)-1)'/fs;

% Process with the SAME functions as preprocessing
adc_vref = 1.1; adc_max = 4095;
emg_proc = EMG_data_processing(emg_adc, fs, adc_vref, adc_max);
imu_proc = IMU_data_processing(imu_raw, fs);

% ---------------- Styling ----------------
figW = 11.0; figH = 3.6; % inches (single-row panels, larger)
LW = 2.2; AXFS = 20; TLW = 1.8;

% ---------------- Save individual panels ----------------
% 1) IMU raw (units only)
save_panel(tIMU, imu_raw, 'Time (s)', 'IMU (m/s^2)', fullfile(plotsDir, 'panel_imu_raw'), figW, figH, LW, AXFS, TLW);
% 2) IMU processed (units only)
save_panel(tIMU, imu_proc, 'Time (s)', 'IMU (m/s^2)', fullfile(plotsDir, 'panel_imu_processed'), figW, figH, LW, AXFS, TLW);
% 3) EMG raw (ADC→V axis as before)
save_panel(tEMG, double(emg_adc)*adc_vref/adc_max, 'Time (s)', 'EMG (V)', fullfile(plotsDir, 'panel_emg_raw'), figW, figH, LW, AXFS, TLW);
% 4) EMG processed (mention Hampel only)
save_panel(tEMG, emg_proc, 'Time (s)', 'EMG (V)', fullfile(plotsDir, 'panel_emg_hampel'), figW, figH, LW, AXFS, TLW);

fprintf('Saved panels in: %s\n', plotsDir);

% ---------------- Helpers (match a_preprocess_data.m) ----------------
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
    v = double(x) * vref / adc_max;            % ADC → volts

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

function [time, fs] = detect_time_and_fs(T, fsDefault)
    time = [];
    fs = fsDefault;
    cand = {'time','Time','timestamp','Timestamp','t','T'};
    for i = 1:numel(cand)
        if ismember(cand{i}, T.Properties.VariableNames)
            time = T.(cand{i});
            break;
        end
    end
    if ~isempty(time)
        try
            time = seconds(time);
        catch
            time = seconds(seconds(time));
        end
        dt = diff(time);
        dt = median(dt(isfinite(dt) & dt>0));
        if ~isempty(dt) && dt > 0
            fs = 1/dt;
        end
    end
end

function save_panel(x, y, xlab, ylab, outBase, figW, figH, LW, AXFS, TLW)
    f = figure('Color','w','Units','inches','Position',[1 1 figW figH]);
    ax = axes(f); hold(ax,'on');
    plot(ax, x, y, 'LineWidth', LW, 'Color', [0.15 0.15 0.15]);
    xlabel(ax, xlab, 'FontName','Helvetica', 'FontSize', AXFS);
    ylabel(ax, ylab, 'FontName','Helvetica', 'FontSize', AXFS);
    set(ax, 'FontName','Helvetica', 'FontSize', AXFS, ...
        'Box','off', 'TickDir','out', 'TickLength',[.015 .015], ...
        'XMinorTick','on', 'YMinorTick','on', 'XGrid','off', 'YGrid','off', ...
        'XColor',[.3 .3 .3], 'YColor',[.3 .3 .3], 'LineWidth', TLW);
    axis(ax, 'tight');
    exportgraphics(f, outBase + ".png", 'Resolution', 600, 'BackgroundColor','white');
    exportgraphics(f, outBase + ".pdf", 'ContentType','vector', 'BackgroundColor','white');
    close(f);
end



