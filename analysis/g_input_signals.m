% Publication-quality individual panels for input signals
clear; clc;

% ---------------- Paths ----------------
scriptFullPath = mfilename('fullpath');
scriptDir = fileparts(scriptFullPath);
projectRoot = fileparts(scriptDir);    % parent of "analysis"
plotsDir   = fullfile(projectRoot, 'plots');
if ~exist(plotsDir, 'dir'); mkdir(plotsDir); end

% ---------------- Files ----------------
% Provide paths to two representative trials from the dataset. Defaults are
% repo-relative examples; adjust to your actual filenames if needed.
file_walk = fullfile(projectRoot, 'data', 'raw', 'Walking forwards', 'Combined_trial92_merged.csv');
file_pick = fullfile(projectRoot, 'data', 'raw', 'Pick up object', 'Combined_trial7_merged.csv');

if ~isfile(file_walk)
    error('file_walk not found: %s\nSet "file_walk" to a valid merged CSV under data/raw/.', file_walk);
end
if ~isfile(file_pick)
    error('file_pick not found: %s\nSet "file_pick" to a valid merged CSV under data/raw/.', file_pick);
end

% ---------------- Styling ----------------
figW = 11.0; figH = 3.6;  % inches
LW = 2.2; AXFS = 18; TLW = 1.8;

%% Panel (a): IMU z-acceleration (walking forwards) - bandpass 0.2–10 Hz
T = readtable(file_walk);
[~, fs] = detect_time_and_fs(T, 1000);
imu = double(T{:, 10});              % column 10
t = (0:numel(imu)-1)'/fs;            % Time (s)
[sos, g] = butter_bandpass(0.2, 10, fs, 5);
imu_f = filtfilt(sos, g, imu);
save_panel(t, imu_f, 'Time (s)', 'IMU (m/s^2)', ...
    fullfile(plotsDir, 'panel_imu_walk_acc'), figW, figH, LW, AXFS, TLW);

%% Panel (b): IMU z-magnitude (squatting) - bandpass 0.2–10 Hz
T = readtable(file_pick);
[~, fs] = detect_time_and_fs(T, 1000);
imu = double(T{:, 13});              % column 13
t = (0:numel(imu)-1)'/fs;
[sos, g] = butter_bandpass(0.2, 10, fs, 5);
imu_f = filtfilt(sos, g, imu);
save_panel(t, imu_f, 'Time (s)', 'IMU (degrees/s)', ...
    fullfile(plotsDir, 'panel_imu_squat_mag'), figW, figH, LW, AXFS, TLW);

%% Panel (c): EMG (walking forwards) - Hampel + bandpass 0.2–400 Hz
T = readtable(file_walk);
[~, fs] = detect_time_and_fs(T, 1000);
emg = double(T{:, 35}) * 1.1/4095;   % column 35 -> volts
t = (0:numel(emg)-1)'/fs;
emg_h = hampel(emg, 50, 3.0);
[sos, g] = butter_bandpass(0.2, 400, fs, 5);
emg_f = filtfilt(sos, g, emg_h);
save_panel(t, emg_f, 'Time (s)', 'EMG (V)', ...
    fullfile(plotsDir, 'panel_emg_walk'), figW, figH, LW, AXFS, TLW);

%% Panel (d): EMG (squatting) - Hampel + bandpass 0.2–400 Hz
T = readtable(file_pick);
[~, fs] = detect_time_and_fs(T, 1000);
emg = double(T{:, 28}) * 1.1/4095;   % column 28 -> volts
t = (0:numel(emg)-1)'/fs;
emg_h = hampel(emg, 50, 3.0);
[sos, g] = butter_bandpass(0.2, 400, fs, 5);
emg_f = filtfilt(sos, g, emg_h);
save_panel(t, emg_f, 'Time (s)', 'EMG (V)', ...
    fullfile(plotsDir, 'panel_emg_squat'), figW, figH, LW, AXFS, TLW);

fprintf('Saved panels in: %s\n', plotsDir);

% ---------------- Helpers ----------------
function [time, fs] = detect_time_and_fs(T, fsDefault)
% Locate a time column (if present) and estimate fs; return [] if none
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

function [sos, g] = butter_bandpass(lowcut, highcut, fs, order)
% Stable bandpass with SOS and gain (for filtfilt)
    nyq = 0.5 * fs;
    Wn  = sort([lowcut, highcut] / nyq);
    Wn(Wn<=0) = eps; Wn(Wn>=1) = 1 - eps;
    [z, p, k] = butter(order, Wn, 'bandpass');
    [sos, g] = zp2sos(z, p, k);
end

function save_panel(x, y, xlab, ylab, outBase, figW, figH, LW, AXFS, TLW)
% Save a single panel as high-res PNG and vector PDF
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



