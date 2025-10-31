<div align="center">

# Exoskeleton: IMU/EMG Data Collection and Motion Intention Modeling

End-to-end pipeline to collect IMU + EMG signals and train/evaluate ML models (CNN, LSTM, SVM) for lower-limb motion intention recognition.

[Dataset (DOI)](https://doi.org/10.17863/CAM.113504) • [Peer‑reviewed article](https://www.nature.com/articles/s41598-025-22103-1)

</div>

## Overview

This repository contains:

- Firmware and notes to configure wearable IMU hardware (EmotiBit/ESP32 based) for data collection.
- MATLAB scripts to preprocess the public dataset, train neural and classical models, and reproduce figures/analyses.
- Analysis utilities to generate confusion matrices, performance tables, and plots.

The dataset used in this work is publicly available via the University of Cambridge Apollo repository: https://doi.org/10.17863/CAM.113504. The full study is published in Nature Scientific Reports: https://www.nature.com/articles/s41598-025-22103-1.

> Note: This repository focuses on data preprocessing and modeling. The live data-collection notebook referenced in earlier notes is not included here.

## Table of contents

- Quick start
- Repository structure
- Dataset
- MATLAB environment and prerequisites
- Reproduce results (end-to-end)
- Scripts guide
- Firmware (optional): flashing EmotiBit/ESP32 via PlatformIO
- Results and figures
- Citation
- License and acknowledgements

## Quick start

1) Get the dataset
- Download from the DOI landing page: https://doi.org/10.17863/CAM.113504
- Unzip to a local folder, e.g., `data/raw/` (create if it doesn’t exist)

2) Open MATLAB
- Open this repo as your MATLAB project, or add it to the path.
- Ensure required toolboxes are available (see below).

3) Run preprocessing
- Edit `src/a_preprocess_data.m` to point to your downloaded data path.
- Run the script to generate processed datasets under `data/processed/`.

4) Train a model
- CNN: run `src/b_run_cnn_model.m`
- LSTM: run `src/c_run_lstm_model.m`
- SVM baseline: run `src/d_run_svm_baseline.m`
- Random baseline: run `src/e_random_baseline.m`

5) Reproduce analyses/plots
- Use scripts in `analysis/` to generate confusion matrices, performance tables, time-series figures, and channel ablation analyses.

## Repository structure

```
analysis/                  % Reproducible analysis and figure scripts (MATLAB)
	a_accuracy_cm.m
	b_confusion_matrix.m
	b_transfer_learning.m
	c_performance_table.m
	d_broken_channel_accuracy.m
	e_timeseries_classification.m
	f_processing_plot.m
	g_input_signals.m

EmotiBit_stock_firmware/   % ESP32/EmotiBit firmware (PlatformIO project)
	platformio.ini
	config.txt               % Example Wi‑Fi config used by the device
	src/                     % Firmware sources
		...

src/                       % MATLAB modeling pipeline
	a_preprocess_data.m      % Point to raw data path; outputs processed features
	b_run_cnn_model.m        % Train/evaluate CNN
	c_run_lstm_model.m       % Train/evaluate LSTM
	d_run_svm_baseline.m     % SVM baseline
	e_random_baseline.m      % Random baseline sanity check
	exoskeleton_library.m    % Shared utility functions
	f_run_transfer_learning.m% Transfer learning experiments
	g_cnn_broken_channels.m  % CNN with channel ablation
	i_timeseries_classification.m % End-to-end time-series classification

README.md
```

## Dataset

- DOI: https://doi.org/10.17863/CAM.113504
- Please review the usage terms on the dataset page. Place the downloaded files under a folder such as `data/raw/`.
- In `src/a_preprocess_data.m`, set the input path to your local raw dataset directory. The script produces processed outputs under `data/processed/` (created if missing).

## MATLAB environment and prerequisites

Recommended toolboxes (depending on which scripts you run):

- Signal Processing Toolbox
- Statistics and Machine Learning Toolbox
- Deep Learning Toolbox

General notes:

- The scripts are standard MATLAB `.m` files and should run on recent MATLAB releases. If you encounter version-specific issues, please open an issue with your MATLAB version and error message.
- Some analyses may take longer on CPU; a GPU (with Deep Learning Toolbox support) will speed up CNN/LSTM training.

## Reproduce results (end-to-end)

1) Preprocess data
- Edit paths in `src/a_preprocess_data.m` to your dataset location.
- Run to generate train/validation/test splits and any derived features used downstream.

2) Train and evaluate models
- CNN: `src/b_run_cnn_model.m` produces training curves and evaluation metrics.
- LSTM: `src/c_run_lstm_model.m` for sequence modeling.
- SVM baseline: `src/d_run_svm_baseline.m` as a classical baseline.
- Random baseline: `src/e_random_baseline.m` sanity check.

3) Analyses and figures
- Confusion matrices and accuracy: `analysis/a_accuracy_cm.m`, `analysis/b_confusion_matrix.m`
- Transfer learning experiments: `analysis/b_transfer_learning.m`, `src/f_run_transfer_learning.m`
- Performance tables: `analysis/c_performance_table.m`
- Broken channel robustness: `analysis/d_broken_channel_accuracy.m`, `src/g_cnn_broken_channels.m`
- Time-series classification figures: `analysis/e_timeseries_classification.m`, `src/i_timeseries_classification.m`
- Processing and input signal plots: `analysis/f_processing_plot.m`, `analysis/g_input_signals.m`

Outputs are written to logical subfolders next to the scripts or into a results directory created by the scripts; check the printed paths in MATLAB’s Command Window.

## Scripts guide

- `src/a_preprocess_data.m`
	- Contract: reads raw dataset, outputs standardized/segmented representations and splits.
	- Edge cases: missing channels, variable sampling rates, empty trials. The script includes checks and will warn if inputs are incomplete.

- `src/b_run_cnn_model.m`
	- Convolutional model for motion intention classification. Accepts the preprocessed dataset; logs metrics and confusion matrices.

- `src/c_run_lstm_model.m`
	- Sequence model for temporal dependencies; good for longer windows.

- `src/d_run_svm_baseline.m`
	- Classical baseline using summary features or flattened windows.

- `src/f_run_transfer_learning.m`
	- Reuses pretrained representations; see comments in the file for the specific source/target configuration.

- `analysis/*`
	- Standalone figure and table generators. They assume model outputs exist; re-run after training to update figures.

## Firmware (optional): EmotiBit/ESP32 via PlatformIO

If you wish to collect your own data using EmotiBit hardware:

1) Install PlatformIO for VS Code
- https://platformio.org/install/ide

2) Open the firmware project
- In VS Code, open `EmotiBit_stock_firmware/` as a PlatformIO project.

3) Configure Wi‑Fi credentials
- Edit or create `EmotiBit_stock_firmware/config.txt` on the SD card used by the device:
	```json
	{"WifiCredentials": [{"ssid": "YOUR_SSID", "password": "YOUR_PASSWORD"}]}
	```
	Use a hotspot (enterprise networks like eduroam typically won’t work for direct device connections).

4) Build and upload
- Connect the board via USB, select the correct COM port, then build and upload from the PlatformIO toolbar.
- For ESP32 WROOM you can also use the EmotiBit Firmware Installer: https://github.com/EmotiBit/EmotiBit_Docs/blob/master/Getting_Started.md

5) EmotiBit software
- Use EmotiBit Oscilloscope/Data Parser per the EmotiBit docs above. Configure UDP output and ensure your host is on the same network.

## Results and figures

The `analysis/` scripts reproduce key tables and figures, including:

- Accuracy and confusion matrices by class
- Transfer learning performance
- Robustness to broken/missing channels
- Example time-series segments, processing and input signal visualizations

Figures are saved to disk by each script; see the script comments for output locations.

## Citation

If you use this repository, the dataset, or the results in your research, please cite:

- Dataset: University of Cambridge Apollo Repository. DOI: https://doi.org/10.17863/CAM.113504
- Article: Nature Scientific Reports. https://www.nature.com/articles/s41598-025-22103-1

Example BibTeX entries (fill in authors/year/title if needed from the landing pages):

```bibtex
@dataset{ruhrberg_exoskeleton_dataset_2025,
	title        = {Exoskeleton IMU/EMG Dataset},
	author       = {Ruhrberg Estevez, Silas and collaborators},
	year         = {2025},
	doi          = {10.17863/CAM.113504},
	url          = {https://doi.org/10.17863/CAM.113504}
}

@article{ruhrberg_exoskeleton_sci_rep_2025,
	title        = {Motion intention recognition for exoskeleton control using IMU and EMG},
	author       = {Ruhrberg Estevez, Silas and collaborators},
	journal      = {Scientific Reports},
	year         = {2025},
	url          = {https://www.nature.com/articles/s41598-025-22103-1}
}
```

## License and acknowledgements

- License: If a license file is not present, this code and documentation are © the authors. Please contact the maintainers for reuse beyond academic purposes.
- Hardware and firmware interfacing relies on the EmotiBit platform: https://emotibit.com and its documentation.
- Thanks to all contributors and participants involved in data collection and validation.

---

Maintainer: Silas Ruhrberg Estevez


