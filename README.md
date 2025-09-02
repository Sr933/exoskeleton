# Code to collect data from IMU and EMG
Author: Silas Ruhrberg Estevez.
Pipeline supports any number of IMUs and EMG recording channels.
Separate script for recording the EMG data.



# 1. Firmware setup of IMU
Skip if IMUs already configured.

For Feather M0 use setup using manually build firmware using Platform I/O
(https://platformio.org/install/ide?install=vscode)

Open folder IMU manual build as project.
Make sure IMU is connected using cable and select the correct COM port at the bottom (USB Device).
Then just press the checkmark to build and upload the firmware.

For ESP-32 WROOM use the EmotiBit automatic Firmware installer (https://github.com/EmotiBit/EmotiBit_Docs/blob/master/Getting_Started.md)

Start Firmware installer and follow instructions on screen.

Finally for both:
Insert SD card with a config.txt file specifying the WiFi name and Password (need to use mobile hotspot not eduroam).

Text in config file:

{"WifiCredentials": [{"ssid": "NAME", "password" : "PASSWORD"}]}

Connect the batteries and make sure they are fully charged.

# 2. Installation of EmotiBit Oscilloscope and Data parser
Follow instructions on Installation of EmotiBit software from:
https://github.com/EmotiBit/EmotiBit_Docs/blob/master/Getting_Started.md

# 3. Data collection setup
Setup hotspot to connect IMUs to (need to edit config.txt file see Firmware section).
Connect computer to the same hotspot as the IMUs.

Open one Oscilloscope tab for each IMU to be used (3).
Select one IMU from the list on the left in each tab.
Select Output stream UDP for each of the tabs.
Check the port number in the Terminal. If not 12346 need to chenge settings of Python script.

# 4. Data collection
Set up EMG collection separately. Files only needed for processing.
Use Exoskeleton data collection Jupyter Notebook.


Verify script is running, may need to pip install the following libraries:
socket
time
threading
os
pandas as pd
csv
numpy as np
shutil

Import the exoskeleton lib and start the data collection code box.

To record a 10s trial press Enter (first and last second are discarded).

Press x to exit the recording session.

# 5. Data processing
Follow instructions in the Jupyter Notebook.

Need EMG recording file and Data Parser installed on system.

Final output is in the Combined files folder: one csv file per trial with 8000 rows each (1kHz sampling for 8s) and 1 column for each channel.


# 6. Machine learning models
Original dataset is preprocessed using the relevant scripts (MATLAB). CNN, SVM and LSTM models can then be used for motion intention in different scenarios. Script produces plots and analysis in respective folders. Users will need to download the raw dataset from Apolloa (https://doi.org/10.17863/CAM.113504) update the link in the preprocessing script.

