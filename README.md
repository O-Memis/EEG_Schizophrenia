# EEG_Schizophrenia
 Schizophrenia - Normal binary classification among adolescent EEG signals, obtained from an open-access dataset. Oguzhan Memis  January 2025

<br><br>



# DATASET DESCRIPTION
<br>

EEG Dataset which contains 2 classes of EEG signals captured from adolescents.
<br><br>
-Classes: Normal (39 people) and Schizophrenia (45 people).
<br><br>
-Properties:
<br><br>    
    16 channels * 128 sample-per-second * 60 seconds of measurement for each person.
<br><br>    
    Voltages are captured in units of microvolts (µV) 10^-6
<br><br><br>


-Orientation:
<br><br>    
    Signals are vertically placed into text files, ordered by channel number (1 to 16).
<br><br>    
    The length of 1 signal is = 128*60 = 7680 samples.
<br><br>
    So each text file contains  16*7680 = 122880 samples, vertically.

<br><br><br><br>
SOURCE:
<br><br>
http://brain.bio.msu.ru/eeg_schizophrenia.htm 
<br><br>
The original article of the dataset:  https://doi.org/10.1007/s10747-005-0042-z  Physiology (Q4) 
<br><br>
A recent article that uses this dataset: https://doi.org/10.1007/s11571-024-10121-0  Cognitive Neuroscience (Q2)

<br><br><br>


**THE CODE IS DIVIDED INTO SEPARATE CELLS, RUN EACH CELL ONE BY ONE CONSECUTIVELY**
<br><br>
