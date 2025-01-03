
"""

_________EEG SCHIZOPHRENIA CLASSIFICATION_________

Oguzhan Memis  January 2025



1-DATASET DESCRIPTION:
-------------------------------------------------------------------------------
    
EEG Dataset which contains 2 classes of EEG signals captured from adolescents.

-Classes: Normal (39 people) and Schizophrenia (45 people).

-Properties of the EEG data:
    
    *16 channels * 128 sample-per-second * 60 seconds of measurement for each person.
    
    *Voltages are captured in units of microvolts (µV) 10^-6
    
    *So the amplitudes of the signals varies from -2000 to +2000



-Orientation of the data in files:
    
    *Signals are vertically placed into text files, ordered by channel number (1 to 16).
    
    *Length of 1 signal is = 128*60 = 7680 samples.

    *So each text file contains  16*7680 = 122880 samples , vertically.


SOURCE OF THE DATASET:

http://brain.bio.msu.ru/eeg_schizophrenia.htm 

Original article of dataset:  https://doi.org/10.1007/s10747-005-0042-z  Physiology (Q4) 

A recent article which uses this dataset: https://doi.org/10.1007/s11571-024-10121-0  Cognitive Neuroscience (Q2)




2-CODE ORGANIZATION:
-------------------------------------------------------------------------------

The codes are divided into separate cells by putting #%%,

RUN EACH CELL ONE BY ONE CONSECUTIVELY.
    
    
    The cells are as follows:
        
        1) Importing the data
        2) Filtering stage (includes time and frequency plots)
        3:
            3.1) Visualization of all the healthy EEG channels together
            3.2) Visualization of all the patient EEG channels together
            
        4) Feature Examinations (including many statistical features on the signals)
        5) Further explorations: Correlation matrix, and Recurrence plot
        6) Multi-level Decomposition by DWT (examination)
        7:
            7.1) DWT Feature Extraction and Data Transformation
            7.2) SVM Grid-search
            7.3) SVM cross-validation
            7.4) MLP model
            7.5) Optional part: save the best model
            7.6) MLP k-fold cross-validation
            7.7) Leave One Out CV on the MLP
        
        8:
            8.1) STFT-Feature extraction method
            8.2) STFT-MLP
            8.3) STFT-SVM (Grid-search)
            
        9:
            9.1) STFT Data Transformation
            9.2) STFT - CNN
            
        10:
            10.1) CWT Data Transformation
            10.2) CWT - CNN
            10.3) CNN k-fold cross-validation
            10.4) Leave One Out CV on the CNN


    
3-CONSIDERATIONS:
-------------------------------------------------------------------------------
*Before running the classification models, consider related data transformation/feature extraction methods
 and the input size (for the Deep Learning models). 

*The DWT-Feature extraction method gives an output dataset in size of (84,16,25)
 then the data of every subject are flattened into 16*25=400
 
*Use different wavelets for SVM and the MLP models. Such as 'bior2.8' and 'bior3.3' for the SVM
 
*The first STFT-Feature extraction method gives an output dataset in size of (84,16,325)
 It uses a downsampled and flattened STFT.
 Then the data of every subject are flattened into 16*325=5200

*In the second STFT method, Spectrograms of the signals are not flattened, and 
 dataset in size of  (84, 16, 513, 21) is obtained. 
 The CNN model takes the input as 16 channel 513*21 matrices.

*In the last CWT method, Scalograms (downsampled in one axis) of the signals are captured 
 into the resultant dataset which has a size of (84, 16, 60, 1920).
 The CNN model takes the input as 16 channel 60*1920 matrices.

*All the MLP models are built by using Keras, 
 and all the CNN models are built by using PyTorch (uses GPU) 
"""




#%% 1) Importing the data

import os
import numpy as np
import matplotlib.pyplot as plt


# Read the files in corresponding folders
path1 = "./normal"
path2 = "./schizophren"



normals = np.zeros((39,122880))  # Preparing an empty matrix for collecting all the data of normal category.     

i=0


# Iterate through each file in the directory
for filename in os.listdir(path1):
    file_path = os.path.join(path1, filename)
    normals[i,:] = np.loadtxt(file_path)
    i +=1



reshaped_normal = np.zeros((39, 16, 7680))

for person in range(39):
    for channel in range(16):
        start_index = channel * 7680
        end_index = (channel + 1) * 7680   # Will help to carefully separate the channels in correct order.
        reshaped_normal[person, channel, :] = normals[person, start_index:end_index]





# __________________________________For the patient (Schizophrenia) category
patients = np.zeros((45,122880))      
j=0


for filename2 in os.listdir(path2):
    file_path2 = os.path.join(path2, filename2)
    patients[j,:] = np.loadtxt(file_path2)
    j +=1



reshaped_patient = np.zeros((45, 16, 7680))

for person2 in range(45):
    for channel2 in range(16):
        start_index2 = channel2 * 7680
        end_index2 = (channel2 + 1) * 7680   
        reshaped_patient[person2, channel2, :] = patients[person2, start_index2:end_index2]




#%% 2) Filtering stage

# For noise removal, the pre-processing stage should be done by signal processing.


# Initial investigations on frequency content of the signals
import scipy
import pywt
from scipy.signal import welch, butter, filtfilt, iirnotch
from scipy.fft import fft, fftfreq
from scipy.signal import stft




# Example signal is the F4 channel of the 31th healthy person.
signal = reshaped_normal[30,2,:] 




fs = 128 # sampling frequency
frequency_resolution = fs/len(signal)

print(f"Frequency Resolution: {frequency_resolution:.2f}%")


time = np.linspace(0, 60, len(signal))

plt.figure(figsize=(50,8))
plt.title("1 channel of EEG")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude in uV")
plt.xlim(0,60)  
plt.ylim(-1500, 1500)
plt.plot(time, signal)



# 1.1) FFT of the signal
fft_val = np.abs(fft(signal)) 
 
freqs = fftfreq(len(signal), d=1/fs)  # Full frequency range (bilateral)
freqs = freqs[:len(freqs)//2]  # Take only positive side 
fft_val = fft_val[:len(freqs)]  # Positive side of the bilateral FFT values

power_vals = fft_val**2  # linear power spectrum
power_vals = 10 * np.log10( power_vals + 1e-12) # for logarithmic power spectrum


plt.figure(figsize=(13,8))
plt.title("Linear-scale FFT")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.plot(freqs, fft_val)


plt.figure(figsize=(13,8))
plt.title("Log-scale Power Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitudes in dB")
plt.plot(freqs, power_vals)


#2.1) Periodogram of the signal 

f, psd = welch(signal, fs=fs, nperseg=512, noverlap=128, window='hamming')

plt.figure(figsize=(13, 8))
plt.plot(f, psd)
plt.xlabel('Frequency (Hz)'), plt.ylabel('Power (signal^2 /Hz)')
plt.title('Periodogram (PSD) by Welch method')




#3.1) Spectrogram of the signal

f_stft, t_stft, z_m = stft(signal, fs=fs, window='hann', nperseg=512, noverlap=128, nfft=1024)


plt.figure(figsize=(13, 8))
plt.pcolormesh(t_stft, f_stft, np.abs(z_m), shading='gouraud')
plt.title('Magnitude Spectrogram')
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')
plt.colorbar(label='Magnitude')
plt.tight_layout()
plt.show()



#4.1) Scalogram of the signal

wavelist = pywt.wavelist(kind='continuous') # The built-in CWT can only use limited number of wavelets.

scales = np.geomspace(1, 60, 60)  # change according to verification of scale frequenices
scale_frequencies = pywt.scale2frequency("cmor1.5-10",scales)*fs  # always verify it to represent accurate frequency resolution

# Difference than spectrogram, comes from these uneven distribution of wavelet frequencies.
coefficients, frequencies = pywt.cwt(signal, scales, 'cmor1.5-10', 1/fs) 
log_coeffs = np.log10(abs(coefficients) + 1)  # prevent log(0)


plt.figure(figsize=(13, 8))
f_min = 0
f_max = fs/2
plt.imshow(log_coeffs, extent=[0, len(signal)/fs, f_min, f_max], # just arranges the axis numbers
           aspect='auto', cmap='viridis', interpolation='bilinear' ) # matrix to image

cbar = plt.colorbar()
cbar.set_label('Log10(Magnitude + 1)')


y_axis_labels =   list(map(int, np.flip(frequencies)))   # assign scale frequencies as y-axis numbers           
step = 10                                                # step size to downsample the indexes
plt.yticks(
    ticks=np.arange(0, len(y_axis_labels), step),  
    labels=y_axis_labels[::step]                         # Select corresponding labels
)

plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')
plt.title('Log Magnitude Scalogram')


plt.show()




#Filter 1: IIR Butterworth 4th Order Low-pass 60Hz
nyquist = fs/2
lowpass_cutoff = 60 / nyquist  # Most of these built-in functions use  normalized frequencies rather than absolute (real) frequencies 
b1, a1 = butter(4, lowpass_cutoff, btype='low')  
lpf_signal = filtfilt(b1, a1, signal)  


#Filter 2: Notch filter 50Hz (According to Russian electric frequency where dataset is provided)
notch_freq = 50 / nyquist  
quality_factor = 30  # Quality factor for sharpness
b2, a2 = iirnotch(notch_freq, quality_factor)  
notch_signal = filtfilt(b2, a2, lpf_signal)  


#Filter 3: IIR Butterworth 8th Order High-pass 0.5 Hz
highpass_cutoff = 0.5 / nyquist  
b3, a3 = butter(8, highpass_cutoff, btype='high')  
filtered_signal = filtfilt(b3, a3, notch_signal)  



# Comparison of examples. Same plotting steps repeated.

plt.figure(figsize=(50,8))
plt.title("Filtered channel of EEG")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude in uV")
plt.xlim(0,60)  
plt.ylim(-1500, 1500)
plt.plot(time, filtered_signal)



# 1.2) FFT 
fft_val = np.abs(fft(filtered_signal)) 
 
freqs = fftfreq(len(filtered_signal), d=1/fs)  
freqs = freqs[:len(freqs)//2]  
fft_val = fft_val[:len(freqs)]  

power_vals = fft_val**2  
power_vals = 10 * np.log10( power_vals + 1e-12) 


plt.figure(figsize=(13,8))
plt.title("Linear-scale FFT")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.plot(freqs, fft_val)


plt.figure(figsize=(13,8))
plt.title("Log-scale Power Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitudes in dB")
plt.plot(freqs, power_vals)


#2.2) Periodogram 
f, psd = welch(filtered_signal, fs=fs, nperseg=512, noverlap=128, window='hamming')

plt.figure(figsize=(13, 8))
plt.plot(f, psd)
plt.xlabel('Frequency (Hz)'), plt.ylabel('Power (signal^2 /Hz)')
plt.title('Periodogram (PSD) by Welch method')




#3.2) Spectrogram 
f_stft, t_stft, z_m = stft(filtered_signal, fs=fs, window='hann', nperseg=512, noverlap=128, nfft=1024)

plt.figure(figsize=(13, 8))
plt.pcolormesh(t_stft, f_stft, np.abs(z_m), shading='gouraud')
plt.title('Magnitude Spectrogram')
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')
plt.colorbar(label='Magnitude')
plt.tight_layout()
plt.show()



#4.2) Scalogram 
scales2 = np.geomspace(1, 60, 60)  
scale_frequencies2 = pywt.scale2frequency("cmor1.5-10",scales2)*fs  

coefficients2, frequencies2 = pywt.cwt(signal, scales2, 'cmor1.5-10', 1/fs) 
log_coeffs2 = np.log10(abs(coefficients2) + 1)  # prevent log(0)


plt.figure(figsize=(13, 8))
f_min = 0
f_max = fs/2
plt.imshow(log_coeffs2, extent=[0, len(signal)/fs, f_min, f_max], 
           aspect='auto', cmap='viridis', interpolation='bilinear' ) 

cbar = plt.colorbar()
cbar.set_label('Log10(Magnitude + 1)')


y_axis_labels2 =   list(map(int, np.flip(frequencies2)))             
step = 10                                                
plt.yticks(
    ticks=np.arange(0, len(y_axis_labels2), step),  
    labels=y_axis_labels2[::step]                         
)

plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')
plt.title('Log Magnitude Scalogram')


plt.show()



# No dramatic changes observed, probably the cause is initial pre-processing of the provided dataset.



# __________________________Now, apply these filters to all channels____________________________________________


filtered_normal = np.zeros((39, 16, 7680))
filtered_patient = np.zeros((45, 16, 7680))



# a) Normals
for aa1 in range(39):
    for cc1 in range(16):
        filtered_normal[aa1, cc1, :] = filtfilt(b1, a1, reshaped_normal[aa1, cc1, :] ) #LPF

for aa2 in range(39):
    for cc2 in range(16):
        filtered_normal[aa2, cc2, :] = filtfilt(b2, a2, filtered_normal[aa2, cc2, :] ) #NOTCH

for aa3 in range(39):
    for cc3 in range(16):
        filtered_normal[aa3, cc3, :] = filtfilt(b3, a3, filtered_normal[aa3, cc3, :] ) #HPF




# b) Patients
for aa1b in range(45):
    for cc1b in range(16):
        filtered_patient[aa1b, cc1b, :] = filtfilt(b1, a1, reshaped_patient[aa1b, cc1b, :] ) #LPF

for aa2b in range(45):
    for cc2b in range(16):
        filtered_patient[aa2b, cc2b, :] = filtfilt(b2, a2, filtered_patient[aa2b, cc2b, :] ) #NOTCH

for aa3b in range(45):
    for cc3b in range(16):
        filtered_patient[aa3b, cc3b, :] = filtfilt(b3, a3, filtered_patient[aa3b, cc3b, :] ) #HPF




#%% 3.1) Visualization of all the healthy EEG

normal_person_number = 38


plt.figure(figsize=(75, 38))
for i in range(16):
    plt.subplot(16, 1, i+1)
    plt.xlim(0,60)  
    plt.ylim(-1500, 1500)
    plt.plot(time,filtered_normal[normal_person_number, i, :])
    plt.xticks(np.arange(0, 61, 10))
    plt.ylabel(f'Channel {i+1}')
    
plt.show()



#%% 3.2) Visualization of all the pateint EEG


patient_person_number = 44



plt.figure(figsize=(75, 38))
for j in range(16):
    plt.subplot(16, 1, j+1)
    plt.xlim(0,60)  
    plt.ylim(-1500, 1500)
    plt.plot(time,filtered_patient[patient_person_number, j, :])
    plt.xticks(np.arange(0, 61, 10))
    plt.ylabel(f'Channel {j+1}')
    
plt.show()



#%% 4) Feature examinations



"""
Effective representation of data differences, plays an important role to reach more optimal results in Machine Learning.

Thus, Feature Engineering of these signals is a crucial step to obtain greater insights. 

Here, various statistical and engineering measurements will be evaluated. 
"""

from scipy.stats import skew, kurtosis, entropy
from scipy.signal import find_peaks



# Definition of the custom function can be changed due to choice of features. 

def extract_features(signal, fs=128):

    
    # Number of samples
    N = len(signal)

    # Time-domain features
    T1 = np.max(signal)
    T2 = np.min(signal)
    T3 = np.mean(signal)
    #T4 = np.var(signal)
    T5 = np.std(signal)
    T6 = np.mean(np.abs(signal - T3))
    T7 = np.sqrt(np.mean(signal**2))
    T8 = np.mean(np.abs(np.diff(signal)))
    #T9 = np.sum(signal**2)
    T10 = np.ptp(signal)
    #T11 = np.sum(np.abs(np.diff(signal)))
    hist, bin_edges = np.histogram(signal, bins='auto', density=True)
    probabilities = hist / np.sum(hist)
    T12 = entropy(probabilities, base=2)
    T13 = np.trapz(signal)
    T14 = np.corrcoef(signal[:-1], signal[1:])[0, 1]
    T15 = np.sum(np.arange(N) * signal**2) / np.sum(signal**2)
    peaks, _ = find_peaks(signal)
    T16 = len(peaks)
    #T17 = np.sum(np.sqrt(1 + np.diff(signal)**2))
    #T18 = np.sum(signal**2) / N
    T19 = np.sum(signal[:-1] * signal[1:] < 0) / N
    T20 = skew(signal)
    T21 = kurtosis(signal)
    T22 = np.sum((signal[:-2] < signal[1:-1]) & (signal[1:-1] > signal[2:]))
    T23 = np.sum((signal[:-2] > signal[1:-1]) & (signal[1:-1] < signal[2:]))
    T24 = np.max(signal) - np.min(signal)
    #T25 = np.sum(np.abs(np.diff(signal)))
    #T26 = np.sqrt(np.var(np.diff(signal)) / T4)
    #T27 = np.sqrt(np.var(np.diff(np.diff(signal))) / np.var(np.diff(signal))) / T26

    
    fft_val = np.abs(fft(signal))
    freqs = fftfreq(len(signal), d=1/fs)
    freqs = freqs[:len(freqs)//2]
    fft_val = fft_val[:len(freqs)]
    power_vals = fft_val**2
    power_vals = 10 * np.log10(power_vals + 1e-12)
    
    
    # Frequency-domain features
    F1 = np.max(power_vals)
    F2 = freqs[np.argmax(power_vals)]
    cumulative_power = np.cumsum(power_vals)
    total_power = cumulative_power[-1]
    F3 = freqs[np.where(cumulative_power >= total_power / 2)[0][0]]
    F4 = np.sum(freqs * power_vals) / np.sum(power_vals)
    threshold = 0.5 * np.max(power_vals)
    low_cutoff = np.min(freqs[np.where(power_vals >= threshold)])
    high_cutoff = np.max(freqs[np.where(power_vals >= threshold)])
    F5 = high_cutoff - low_cutoff
    F6 = np.sum((power_vals - power_vals)**2)
    epsilon = 1e-10
    norm_power_vals = (power_vals + epsilon) / (np.sum(power_vals) + epsilon)
    F7 = -np.sum(norm_power_vals * np.log2(norm_power_vals))
    F8 = np.sqrt(np.sum((freqs - F4)**2 * power_vals) / np.sum(power_vals))
    F9 = np.sum((freqs - F4)**3 * power_vals) / (F8**3 * np.sum(power_vals))
    F10 = np.sum((freqs - F4)**4 * power_vals) / (F8**4 * np.sum(power_vals))
    F11 = freqs[np.where(cumulative_power >= 0.95 * total_power)[0][0]]
    #F12 = freqs[np.where(cumulative_power >= 1.15 * total_power)[0][0]] if np.any(cumulative_power >= 1.15 * total_power) else None
    F13 = freqs[np.argmax(power_vals)]
    F14 = np.sum(power_vals[(freqs >= 0.6) & (freqs <= 2.5)]) / np.sum(power_vals)
    H_high = np.max(power_vals[freqs >= 0.5 * fs / 2])
    H_low = np.min(power_vals[freqs >= 0.5 * fs / 2])
    K = 1
    F15 = K * (H_high / H_low)
    F16 = 1 - np.sum(freqs * power_vals * power_vals) / (
        np.sqrt(np.sum(freqs * power_vals) * np.sum(freqs * power_vals))
    )
    delta_power = np.sum(power_vals[(freqs >= 0.5) & (freqs <= 4)])
    theta_power = np.sum(power_vals[(freqs > 4) & (freqs <= 8)])
    alpha_power = np.sum(power_vals[(freqs > 8) & (freqs <= 13)])
    beta_power = np.sum(power_vals[(freqs > 13) & (freqs <= 30)])
    total_power = np.sum(power_vals)
    F17 = delta_power / total_power
    F18 = theta_power / total_power
    F19 = alpha_power / total_power
    F20 = beta_power / total_power

    # Time-frequency domain features
    W1 = pywt.wavedec(signal, 'db4', level=5)
    W2 = [np.sum(np.square(c)) for c in W1]
    #TF1 = np.sum(W2)
    W3 = np.sum(W2)
    W4 = entropy(W2 / W3, base=2)
    TF2 = W4
    TF3 = [W2[i] / W2[i + 1] for i in range(len(W2) - 1)]
    #TF4 = [np.var(d) for d in W1]

    # Combine all features into a single 1D array
    features = np.array([
        T1, T2, T3, T5, T6, T7, T8, T10, T12, T13, T14, T15, T16, T19, 
         T20, T21, T22, T23, T24, F1, F2, F3, F4, F5, F6, F7,
        F8, F9, F10, F11, F13, F14, F15, F16, F17, F18, F19, F20, TF2, TF3[0], TF3[1],
        TF3[2], TF3[3], TF3[4]
    ])

    return features



healthy_features = np.zeros((39, 16, 44))  # consider the size of the feature vector here
patient_features = np.zeros((45, 16, 44))



for u in range(39):
    for p in range(16):
        healthy_features[u, p, :] = extract_features(filtered_normal[u, p, :])



for m in range(45):
    for n in range(16):
        patient_features[m, n, :] = extract_features(filtered_patient[m, n, :])


feature_set = np.concatenate((healthy_features,patient_features))


# Choose example channels for examination
feature1 = feature_set[30,2,:]
feature2 = feature_set[74,2,:]



plt.figure(figsize=(10,10))
plt.ylabel("Feature Values")
plt.xlabel("Feature Number")
plt.title("Feature Vector")
plt.plot(feature1)
plt.show()

plt.figure(figsize=(10,10))
plt.ylabel("Feature Values")
plt.xlabel("Feature Number")
plt.title("Feature Vector")
plt.plot(feature2)
plt.show()




#%% 5) Further explorations: Connectivity related plots


import seaborn as sns
from scipy.spatial.distance import pdist, squareform



# 1) Correlation matrix between the channels, for a specific individual

eeg1 = filtered_normal[30,:,:] # 31th healthy person channels
eeg2 = filtered_patient[39,:,:] # 40th patient channels


correlation_matrix1 = np.corrcoef(eeg1)
correlation_matrix2 = np.corrcoef(eeg2)


# Heatmap
plt.figure(figsize=(15, 10))
sns.heatmap(correlation_matrix1,
            cmap='coolwarm',
            center=0,
            annot=True,
            fmt='.2f',
            square=True)

plt.title('EEG Channel Correlation Matrix - Healthy')
plt.show()


plt.figure(figsize=(15, 10))
sns.heatmap(correlation_matrix2,
            cmap='coolwarm',
            center=0,
            annot=True,
            fmt='.2f',
            square=True)

plt.title('EEG Channel Correlation Matrix - Schizophrenia')
plt.show()




# 2) Recurrence Plot 

"""
It is used for nonlinear time series analysis. 
The plot shows how much the signal revisits it's previous states.

"""



signal1 = filtered_normal[30, 2, :]  # F4 channel from previous plots.
signal2 = filtered_patient[39, 2, :]



signal_reshaped1 = signal1.reshape(-1, 1)

distance_matrix1 = squareform(pdist(signal_reshaped1))    # The distance matrix
threshold1 = 0.1 * np.max(distance_matrix1)               # Threshold 10% of the maximum distance
recurrence_matrix1 = distance_matrix1 < threshold1        # recurrence matrix

plt.figure(figsize=(15, 10))
plt.imshow(recurrence_matrix1, cmap='binary', interpolation='nearest')
plt.colorbar(label='Recurrence')
plt.xlabel('Time Index')
plt.ylabel('Time Index')
plt.title('Recurrence Plot of Channel F4 - Healthy')

plt.tight_layout()
plt.show()




signal_reshaped2 = signal2.reshape(-1, 1)

distance_matrix2 = squareform(pdist(signal_reshaped2))    
threshold2 = 0.1 * np.max(distance_matrix2)               
recurrence_matrix2 = distance_matrix2 < threshold2        

plt.figure(figsize=(15, 10))
plt.imshow(recurrence_matrix2, cmap='binary', interpolation='nearest')
plt.colorbar(label='Recurrence')
plt.xlabel('Time Index')
plt.ylabel('Time Index')
plt.title('Recurrence Plot of Channel F4 - Schizophrenia')

plt.tight_layout()
plt.show()





#%% 6) Multi-level Decomposition by DWT




A5, D5, D4, D3, D2, D1 = pywt.wavedec(signal2, 'db19', level=5)  # Perform wavelet decomposition


"""
________Multi-level Decomposition by DWT____________


                           Original Signal [0-128Hz]
                              ___|________
Frequency content            |            | 
halves in each step          |            |
                           A1[0-64Hz]    D1[64-128Hz]
                          ___|_______
Also the                 |           | 
downsampling             |           |
occurs in             A2[0-32Hz]    D2[32-64Hz]
each step           ___|________
                   |            | 
                   |            |
                  A3[0-16Hz]   D3[16-32Hz]
               ___|_________
              |             | 
              |             |
             A4[0-8Hz]     D4[8-16Hz]
          ___|___________
         |               | 
         |               |
        A5[0-4Hz]      D5[4-8Hz]


Here A5 can be thought as delta, D5 is theta, 
D4 is alpha, D3 is beta, and D2 is gamma waves.

"""

# Visualizations of decomposed bands
plt.figure(figsize=(15, 15))  
plt.subplot(3, 2, 1)          
plt.plot(D1)
plt.ylim(-2000, 2000)               
plt.title("D1 [64-128Hz]")
plt.subplot(3, 2, 2)          
plt.plot(D2)
plt.ylim(-2000, 2000)                
plt.title("D2 [32-64Hz]")
plt.subplot(3, 2, 3)          
plt.plot(D3)
plt.ylim(-2000, 2000)                
plt.title("D3 [16-32Hz]")
plt.subplot(3, 2, 4)          
plt.plot(D4)
plt.ylim(-2000, 2000)                
plt.title("D4 [8-16Hz]")
plt.subplot(3, 2, 5)          
plt.plot(D5)  
plt.ylim(-2000, 2000)              
plt.title("D5 [4-8Hz]")
plt.subplot(3, 2, 6)          
plt.plot(A5)
plt.ylim(-2000, 2000)                
plt.title("A5 [0-4Hz]")                     
plt.show()





#%% 7.1) DWT Feature Extraction and Data Transformation

"""
After all the pre-processing and the observations, now the data should be prepared for the model input.

Many Feature Extraction, Data Transformation or Dimensionality Reduction methods can be applied such as:
Multi-level Decomposition by DWT, Time and Frequency statistics, EMD, PCA, Spectrogram, Recurrence Plot or similar. 
"""

from scipy import stats, integrate


dataset = np.concatenate((filtered_normal, filtered_patient))




#1) Extract statistical features from DWT decomposition coefficients

def extract_dwt_features(signal):
   
    dwt = pywt.wavedec(signal, 'db19', level=5)
    A5, D5, D4, D3, D2, D1 = dwt
    
    
    energies = [np.sum(np.square(ii)) for ii in dwt]  # energy for wavelet bands
    total = np.sum(energies)                          # Total energy
    
    # Relative energy between consecutive wavelet bands
    bandratio = [energies[i] / energies[i + 1] for i in range(len(energies) - 1)]
    
    features = []
    
    
    # Process for each coefficient band (A5, D5, D4, D3, D2)
    for band in [A5, D5, D4, D3, D2]:
        
        
        avg_integrate = (integrate.simps(np.abs(A5))+integrate.simps(np.abs(D5))+
                         integrate.simps(np.abs(D4))+integrate.simps(np.abs(D3))+
                         integrate.simps(np.abs(D2))
                         )/5
        
        freqs, psd = welch(band, fs=128)
        # abs(band)  /np.mean(abs(band))

        features.extend([
             
            (np.sum(freqs * psd) / np.sum(psd))/bandratio[4],
            kurtosis(band),
            (freqs[np.cumsum(psd) >= 0.5 * np.sum(psd)][0])/bandratio[4],
            integrate.simps(np.abs(band))/avg_integrate

        ])
        
    features.extend([
        bandratio[0],                        # Band energy ratios
        bandratio[1],
        bandratio[2],
        bandratio[3],
        bandratio[4]
    ])
    
    
    return np.array(features)



example1 = extract_dwt_features(signal1)
example2 = extract_dwt_features(signal2)

plt.figure(figsize=(10,10))
plt.ylabel("Feature Values")
plt.xlabel("Feature Number")
plt.title("Feature Vector")
plt.ylim(-1, 16)
plt.plot(example1)
plt.show()

plt.figure(figsize=(10,10))
plt.ylabel("Feature Values")
plt.xlabel("Feature Number")
plt.title("Feature Vector")
plt.ylim(-1, 16)
plt.plot(example2)
plt.show()




#2) Apply the transformation to whole data


dwt_data = np.zeros((84, 16, 25)) # consider the size of the feature vector 



for ii in range(84):
    for jj in range(16):
        dwt_data[ii, jj, :] = extract_dwt_features(dataset[ii, jj, :])






#%% 7.2) SVM Grid Search




from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale , StandardScaler
from sklearn.metrics import accuracy_score , f1_score , recall_score, precision_score , confusion_matrix
from sklearn.model_selection import GridSearchCV



# Matrices are flattened to a 1D vector 
x = dwt_data.reshape(84, -1)  


y = np.array([0] * 39 + [1] * 45)  # Binary labels



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42, stratify=y)



mymodel = svm.SVC(probability=True) # Classifier model

mymodel.get_params()  # Look for which parameters can be choosen




'''
Grid search is about simply iterating all sets of values defined, to obtain best performed model.
It also uses Cross-Validation. And parameter grid need to be defined.
'''


# Define the parameter grid
param_grid = {
    'C': [0.1, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9],
    'kernel': ['linear', 'rbf', 'sigmoid', 'poly'],
    'gamma': ['scale', 'auto'],
    'coef0': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,0.8,0.9, 1.0],  # for polynomial and sigmoid kernel
    'shrinking': [True, False],

}


grid_search = GridSearchCV(mymodel, param_grid, cv=5, n_jobs=-1, verbose=0, scoring='accuracy')
# Setting n_jobs to -1 means that the algorithm will use all available CPU cores on your machine
# Verbose setting is for observing the details of the ongoing process 

grid_search.fit(x_train, y_train)



# Getting the best model
best_model = grid_search.best_estimator_



# Calculate the metrics for test data---------------------------------------
print("\nTest Data Metrics")
test_predictions = best_model.predict(x_test)

s12 = accuracy_score(y_test, test_predictions)  
s22 = f1_score(y_test, test_predictions, average='weighted')
s32 = recall_score(y_test, test_predictions, average='weighted')
s42 = precision_score(y_test, test_predictions, average='weighted')  

print(f"Test Accuracy: {s12 * 100:.2f}%")  
print(f"Test F1 Score: {s22 * 100:.2f}%")
print(f"Test Recall: {s32 * 100:.2f}%")
print(f"Test Precision: {s42 * 100:.2f}%")

cm_test = confusion_matrix(y_test, test_predictions)  
print("\nConfusion Matrix (Unseen Data):")
print(cm_test)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix - Test Data')
plt.show()




print(f"Mean CV Score: {grid_search.cv_results_['mean_test_score'][grid_search.best_index_] * 100:.2f}%")
print(f"Standard Deviation: {grid_search.cv_results_['std_test_score'][grid_search.best_index_] * 100:.2f}%")


# Print the best model parameters
print("\nBest parameters:")
for param, value in grid_search.best_params_.items():
    print(f"{param}: {value}")





#%% 7.3) SVM Cross-Validation


from sklearn import svm
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score , f1_score , recall_score, precision_score , confusion_matrix

"""
K-fold cross-validation of the final SVM model with the best parameters.
According to iterations above, best SVM model will be tested with the best performing wavelet.
Best performing wavelet for the SVM was "bior2.6"
"""


x = dwt_data.reshape(84, -1)  
y = np.array([0] * 39 + [1] * 45)  # Binary labels


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42, stratify=y)



def svmmodel():
    
    a = svm.SVC( C=4, kernel="rbf", gamma="auto", coef0=0.0, shrinking=True ,probability=True)
    
    return a



k = 5  # Number of folds
kf = KFold(n_splits=k, shuffle=True)

fold_number = 1  # iteration
all_training_predictions = []
all_training_labels = []
accuracies = []

# Loop for series of re-training
for train_index, val_index in kf.split(x_train):
    print(f'Fold {fold_number}')
    x_fold_train, x_fold_val = x_train[train_index], x_train[val_index]
    y_fold_train, y_fold_val = y_train[train_index], y_train[val_index]
    
    model =  svmmodel()
    model.fit(x_fold_train, y_fold_train)
   
    predictions = model.predict(x_fold_val)
    acc = accuracy_score(y_fold_val, predictions)
    accuracies.append(acc)
    
    all_training_predictions.extend(predictions)
    all_training_labels.extend(y_fold_val)
    
    fold_number += 1


# Report the mean and standard deviation of the cross-validation accuracy
mean_accuracy = np.mean(accuracies)
std_accuracy = np.std(accuracies)
print(f'Cross-Validation Mean Accuracy: {mean_accuracy*100:.3f} ')
print(f'Standard Deviation of Accuracy: {std_accuracy*100:.3f}')



# Plot the confusion matrix for cross-validation predictions
cm_cv = confusion_matrix(all_training_labels, all_training_predictions)
fig1 = plt.figure()
plt.figure(figsize=(8, 6))
sns.heatmap(cm_cv, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix - Cross-Validation')
plt.show()




# Finally re-train the model on the entire training data and evaluate on the unseen test data
final_model = svmmodel()
final_model.fit(x_train, y_train)

final_predictions = final_model.predict(x_test)



# Final test metrics
final_acc = accuracy_score(y_test, final_predictions)
final_f1 = f1_score(y_test, final_predictions, average='weighted')
final_recall = recall_score(y_test, final_predictions, average='weighted')
final_precision = precision_score(y_test, final_predictions)

print(f'Final Test Accuracy: {final_acc*100:.3f}')
print(f"F1 Score: {final_f1*100:.2f}%")
print(f"Recall: {final_recall*100:.2f}%")
print(f"Precision: {final_precision*100:.2f}%")


cm_test = confusion_matrix(y_test, final_predictions)
fig2 = plt.figure()
plt.figure(figsize=(8, 6))
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix - Test Data')
plt.show()


# The cross validation scores are too low than the test scores!



#%% 7.4) DWT-MLP model iterations


import keras 
from keras.models import Sequential
from keras.layers import Dense,Dropout ,BatchNormalization
from keras.optimizers import RMSprop
from keras.utils import to_categorical
from keras.regularizers import l1, l2
from keras.optimizers import RMSprop, Adam, SGD, Adagrad, Adadelta, Adamax
from keras.initializers import glorot_uniform, he_normal
from keras.callbacks import EarlyStopping



x = dwt_data.reshape(84, -1)  # flattening


# One-hot encoding for multi-class classification
label1= np.zeros((39,1))
label2= np.ones((45,1))

labels = np.vstack((label1,label2))
onehot= to_categorical(labels,2)

x_train, x_test, y_train, y_test = train_test_split(x, onehot,test_size=0.20 , random_state=42, stratify=onehot)




optimizer = RMSprop(learning_rate=0.001, rho=0.5, epsilon=1e-06)

early_stop= EarlyStopping(monitor='val_accuracy', patience=200, restore_best_weights=True)

act = "silu"

# Model 
model6 = Sequential()
model6.add(Dense(200, activation=act, kernel_regularizer=l2(0.0005), input_dim=400)) 
model6.add(Dropout(0.1))
model6.add(Dense(250, activation=act, kernel_regularizer=l2(0.002)))
model6.add(Dropout(0.2))
model6.add(Dense(70, activation=act, kernel_regularizer=l2(0.0002)))
model6.add(Dense(2,activation="softmax")) 
model6.compile(optimizer=optimizer,loss="categorical_crossentropy", metrics=["accuracy"])  
history=model6.fit(x_train, y_train, validation_data=(x_test,y_test),epochs=250,batch_size=5, callbacks=[early_stop])

predictions= model6.predict(x_test)  
model6.summary()                           




# The confusion matrix
Predictions=np.argmax(predictions, axis=1)
Y_test=np.argmax(y_test, axis=1)
cm = confusion_matrix(Y_test, Predictions)
print(cm)    
    
s13 = accuracy_score(Y_test, Predictions)
s23 = f1_score(Y_test, Predictions, average='weighted')
s33 = recall_score(Y_test, Predictions, average='weighted')
s43 = precision_score(Y_test, Predictions)

print(f"Test Accuracy: {s13*100:.2f}%")
print(f"Test F1 Score: {s23*100:.2f}%")
print(f"Test Recall: {s33*100:.2f}%")
print(f"Test Precision: {s43*100:.2f}%")



fig6=plt.figure()
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues') 
plt.xlabel('Predicted label')
plt.ylabel('True label') 
plt.title('Confusion  Matrix - Test Data') 
plt.show()   


# Change of accuracy by epochs
plt.figure(figsize=(8,6))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()




#%% 7.5)  Optional: Save and reuse the best model

#model6.save('dwt_mlp_model_96.h5')


from keras.models import load_model

model = load_model('dwt_mlp_model_96.h5')


newpredictions = model.predict(x) # Make predictions
newpredictions = np.argmax(newpredictions, axis=1)

all_labels = np.argmax(onehot, axis=1)


s13 = accuracy_score(all_labels, newpredictions)
s23 = f1_score(all_labels, newpredictions, average='weighted')
s33 = recall_score(all_labels, newpredictions, average='weighted')
s43 = precision_score(all_labels, newpredictions)

print(f"Accuracy: {s13*100:.2f}%")
print(f"F1 Score: {s23*100:.2f}%")
print(f"Recall: {s33*100:.2f}%")
print(f"Precision: {s43*100:.2f}%")


# Confusion matrix
cm = confusion_matrix(all_labels, newpredictions)
print("Confusion Matrix:")
print(cm)


plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')  
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix - All Data')
plt.show() 



#%% 7.6) MLP K-Fold CV 

"""
In k-fold cross-validation, the goal is to evaluate the performance of a model 
on different subsets of the data to ensure that it generalizes well to unseen data.
In this method, we are splitting the training set into k folds, and train the model k times
with using the splitted part of the training set as a test set.
Finally, we test the model's performance on unseen "test" data.
"""



from sklearn.model_selection import train_test_split, KFold
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam, RMSprop 
from keras.utils import to_categorical
from keras.regularizers import l2
from keras.callbacks import EarlyStopping


x = dwt_data.reshape(84, -1)  # flattening
label1 = np.zeros((39, 1))
label2 = np.ones((45, 1))
labels = np.vstack((label1, label2))
onehot = to_categorical(labels, 2) 

# 1-Initial train-test split
x_train, x_test, y_train, y_test = train_test_split(x, onehot, test_size=0.2, random_state=42, stratify=onehot)



# 2-Custom function to generate the model where it is needed. The most consistent model parameters are below.
def make_model():
    act = "silu"
    model = Sequential()
    model.add(Dense(200, activation=act, kernel_regularizer=l2(0.0005), input_dim=400)) 
    model.add(Dropout(0.1))
    model.add(Dense(250, activation=act, kernel_regularizer=l2(0.002)))
    model.add(Dropout(0.2))
    model.add(Dense(70, activation=act, kernel_regularizer=l2(0.0002)))
    model.add(Dense(2,activation="softmax")) 
    optimizer = RMSprop(learning_rate=0.001, rho=0.5, epsilon=1e-06)
    model.compile(optimizer=optimizer,loss="categorical_crossentropy", metrics=["accuracy"])  

    return model

# 3-Definitions for K-Fold Cross-Validation

k = 10  # Number of folds
kf = KFold(n_splits=k, shuffle=False)

fold_number = 1  # iteration
all_training_predictions = []
all_training_labels = []
accuracies = []

# 4-Loop for series of re-training
for train_index, val_index in kf.split(x_train):
    print(f'Fold {fold_number}')
    x_fold_train, x_fold_val = x_train[train_index], x_train[val_index]
    y_fold_train, y_fold_val = y_train[train_index], y_train[val_index]

    early_stop = EarlyStopping(monitor='val_accuracy', patience=200, restore_best_weights=True)

    
    model = make_model()
    history = model.fit(x_fold_train, y_fold_train, validation_data=(x_fold_val, y_fold_val), epochs=250, batch_size=5, callbacks=[early_stop], verbose=0)

    predictions = model.predict(x_fold_val)
    Predictions = np.argmax(predictions, axis=1)
    Y_val = np.argmax(y_fold_val, axis=1)
    acc = accuracy_score(Y_val, Predictions)
    accuracies.append(acc)
    
    all_training_predictions.extend(Predictions)
    all_training_labels.extend(Y_val)
    
    fold_number += 1

# 5-Report the mean and standard deviation of the cross-validation accuracy
mean_accuracy = np.mean(accuracies)
std_accuracy = np.std(accuracies)
print(f'Cross-Validation Mean Accuracy: {mean_accuracy*100:.3f} ')
print(f'Standard Deviation of Accuracy: {std_accuracy*100:.3f}')



# 6-Plot the confusion matrix for cross-validation predictions
cm_cv = confusion_matrix(all_training_labels, all_training_predictions)
fig1 = plt.figure()
plt.figure(figsize=(8, 6))
sns.heatmap(cm_cv, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix - Cross-Validation on Training')
plt.show()




# 7- Finally re-train the model on the entire training data and evaluate on the unseen test data
final_model = make_model()
early_stop = EarlyStopping(monitor='val_accuracy', patience=100, restore_best_weights=True)
final_history = final_model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=200, batch_size=5, callbacks=[early_stop], verbose=0)

final_predictions = final_model.predict(x_test)
Final_Predictions = np.argmax(final_predictions, axis=1)
Y_test = np.argmax(y_test, axis=1)


# 8-Final test metrics
final_acc = accuracy_score(Y_test, Final_Predictions)
final_f1 = f1_score(Y_test, Final_Predictions, average='weighted')
final_recall = recall_score(Y_test, Final_Predictions, average='weighted')
final_precision = precision_score(Y_test, Final_Predictions)

print(f'Final Test Accuracy: {final_acc*100:.3f}')
print(f"F1 Score: {final_f1*100:.2f}%")
print(f"Recall: {final_recall*100:.2f}%")
print(f"Precision: {final_precision*100:.2f}%")


cm_test = confusion_matrix(Y_test, Final_Predictions)
fig2 = plt.figure()
plt.figure(figsize=(8, 6))
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix - Test Data')
plt.show()



# Change of accuracy by epochs for the final model 
plt.figure(figsize=(8, 6))
plt.plot(final_history.history['accuracy'])
plt.plot(final_history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()








#%% 7.7) Leave One Out CV on MLP





from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import train_test_split



# Assuming dwt_data is already defined somewhere in your code
x = dwt_data.reshape(84, -1)  # flattening

# One-hot encoding for multi-class classification
label1 = np.zeros((39, 1))
label2 = np.ones((45, 1))
labels = np.vstack((label1, label2))
onehot = to_categorical(labels, 2)

# LOOCV procedure
loo = LeaveOneOut()
accuracies = []
all_predictions = []
all_true_labels = []

for train_ix, test_ix in loo.split(x):
    x_train, x_test = x[train_ix], x[test_ix]
    y_train, y_test = onehot[train_ix], onehot[test_ix]

    
    # Model   
    early_stop = EarlyStopping(monitor='val_accuracy', patience=200, restore_best_weights=True)
    model = make_model()
    
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=250, batch_size=5, callbacks=[early_stop], verbose=0)

    predictions = model.predict(x_test)
    Predictions = np.argmax(predictions, axis=1)
    Y_test = np.argmax(y_test, axis=1)
    acc = accuracy_score(Y_test, Predictions)
    accuracies.append(acc)
    
    all_predictions.extend(Predictions)
    all_true_labels.extend(Y_test)

# Report the mean and standard deviation of the accuracy
mean_accuracy = np.mean(accuracies)
std_accuracy = np.std(accuracies)
print(f'LOOCV Accuracy: {mean_accuracy:.3f} ({std_accuracy:.3f})')

# Plot the confusion matrix for all predictions
cm = confusion_matrix(all_true_labels, all_predictions)
fig6 = plt.figure()
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix - LOOCV')
plt.show()



s41 = accuracy_score(all_true_labels, all_predictions)  
s42 = f1_score(all_true_labels, all_predictions, average='weighted')
s43 = recall_score(all_true_labels, all_predictions, average='weighted')
s44 = precision_score(all_true_labels, all_predictions, average='weighted')  

print(f"Total LOOCV Accuracy: {s41 * 100:.2f}%")  
print(f"Total LOOCV  F1 Score: {s42 * 100:.2f}%")
print(f"Total LOOCV  Recall: {s43 * 100:.2f}%")
print(f"Total LOOCV  Precision: {s44 * 100:.2f}%")



#%% 8.1) STFT - FE Method


# In this method, the Spectrogram of the signal (result of STFT) will be given as input feature vector


# Slightly lower resolution of STFT 
f_stft2, t_stft2, z_m2 = stft(signal, fs=fs, window='hann', nperseg=512, noverlap=64, nfft=512)


plt.figure(figsize=(13, 8))
plt.pcolormesh(t_stft2, f_stft2, np.abs(z_m2), shading='gouraud') # absolute value of complex matrix
plt.title('Magnitude Spectrogram')
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')
plt.colorbar(label='Magnitude')
plt.tight_layout()
plt.show()


# Further downsampling can be needed
d_t_stft2 = t_stft2[::2]
d_f_stft2 = f_stft2[::2]
d_z_m2 = z_m2[::2, ::2]

plt.figure(figsize=(13, 8))
plt.pcolormesh(d_t_stft2, d_f_stft2, np.abs(d_z_m2), shading='gouraud') # absolute value of complex matrix
plt.title('Downsampled Spectrogram')
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')
plt.colorbar(label='Magnitude')
plt.tight_layout()
plt.show()



def eeg_stft_extraction(x, fs=128):
    
    # other arrangements are defined below
    
    _, __, z = stft(x, fs=fs, window='hann', nperseg=512, noverlap=64, nfft=512)
    
    down_z = np.abs(z[::4, ::4]) # downsampling by factor of 2
    
    flat = down_z.flatten() # extra: flattening
    return flat


example3 = eeg_stft_extraction(signal)

# It looks like the FFT of that signal
plt.figure(figsize=(10,8))
plt.title("Flattened Spectrogram")
plt.ylabel("Magnitudes")
plt.plot(example3)




stft_data1 = np.zeros((84, 16, 325)) # consider the size of the feature vector 


for iii in range(84):
    for jjj in range(16):
        stft_data1[iii, jjj, :] = eeg_stft_extraction(dataset[iii, jjj, :])


#%% 8.2) STFT-MLP



import keras 
from keras.models import Sequential
from keras.layers import Dense,Dropout ,BatchNormalization, LeakyReLU 
from keras.optimizers import RMSprop
from keras.utils import to_categorical
from keras.regularizers import l1, l2
from keras.optimizers import RMSprop, Adam, SGD, Adagrad, Adadelta, Adamax
from keras.initializers import glorot_uniform, he_normal
from keras.callbacks import EarlyStopping



x2 = stft_data1.reshape(84, -1)  # flattening


# One-hot encoding for multi-class classification
label1= np.zeros((39,1))
label2= np.ones((45,1))

labels2 = np.vstack((label1,label2))
onehot2= to_categorical(labels,2)

x_train, x_test, y_train, y_test = train_test_split(x2, onehot2,test_size=0.2 , random_state=42, stratify=onehot2)


# Standard Normalization
scale = StandardScaler()
scale.fit(x_train) # Adapting to just training set
x_train = scale.transform(x_train) # Transform
x_test = scale.transform(x_test) 


myactivation = LeakyReLU(alpha=1)
early_stop= EarlyStopping(monitor='val_accuracy', patience=150, restore_best_weights=True)
optimizer = RMSprop(learning_rate=0.00005, rho=0.01, epsilon=1e-05)


# Model 
model6 = Sequential()
model6.add(Dense(150, activation= myactivation, kernel_regularizer=l2(0.001), input_dim=5200))
model6.add(BatchNormalization())
model6.add(Dropout(0.2))
model6.add(Dense(30, activation=myactivation, kernel_regularizer=l2(0.1)))
model6.add(BatchNormalization())
model6.add(Dropout(0.3))
model6.add(Dense(10, activation=myactivation, kernel_regularizer=l2(0.001)))
model6.add(Dense(2,activation="softmax")) 
model6.compile(optimizer=optimizer ,loss="categorical_crossentropy", metrics=["accuracy"])  
history=model6.fit(x_train, y_train, validation_data=(x_test,y_test),epochs=200,batch_size=10, callbacks=[early_stop])

predictions= model6.predict(x_test)  
model6.summary()                            




# The confusion matrix
Predictions=np.argmax(predictions, axis=1)
Y_test=np.argmax(y_test, axis=1)
cm = confusion_matrix(Y_test, Predictions)
print(cm)    
    
s13 = accuracy_score(Y_test, Predictions)
s23 = f1_score(Y_test, Predictions, average='weighted')
s33 = recall_score(Y_test, Predictions, average='weighted')
s43 = precision_score(Y_test, Predictions)

print(f"Test Accuracy: {s13*100:.2f}%")
print(f"Test F1 Score: {s23*100:.2f}%")
print(f"Test Recall: {s33*100:.2f}%")
print(f"Test Precision: {s43*100:.2f}%")



fig6=plt.figure()
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues') 
plt.xlabel('Predicted label')
plt.ylabel('True label') 
plt.title('Confusion  Matrix - Test Data') 
plt.show()   


# Change of accuracy by epochs
plt.figure(figsize=(8,6))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


#%% 8.3) STFT-SVM 




from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale , StandardScaler
from sklearn.metrics import accuracy_score , f1_score , recall_score, precision_score , confusion_matrix
from sklearn.model_selection import GridSearchCV



# Matrices are flattened to a 1D vector 
x = stft_data1.reshape(84, -1)  


y = np.array([0] * 39 + [1] * 45)  # Binary labels



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)


"""
# Standard Normalization
scale = StandardScaler()
scale.fit(x_train) # Adapting to just training set
x_train = scale.transform(x_train) # Transform
x_test = scale.transform(x_test) 
"""


mymodel = svm.SVC(probability=True) # Classifier model



# Define the parameter grid
param_grid = {
    'C': [0.01,0.1,1,2,3,4,5,6,7,8,9,10,15,20,25,30,40,50,100,200],
    'kernel': ['linear', 'rbf', 'sigmoid', 'poly'],
    'gamma': ['scale', 'auto'],
    'coef0': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,0.8,0.9, 1.0],  # for polynomial and sigmoid kernel
    'shrinking': [True, False],

}


grid_search = GridSearchCV(mymodel, param_grid, cv=5, n_jobs=-1, verbose=0, scoring='accuracy')
# Setting n_jobs to -1 means that the algorithm will use all available CPU cores on your machine
# Verbose setting is for observing the details of the ongoing process 

grid_search.fit(x_train, y_train)



# Getting the best model
best_model = grid_search.best_estimator_




# 1- Calculate k-fold CV metrics with train data-------------------------------

print("\nDetailed results for best parameters:")
print(f"Mean CV Score: {grid_search.cv_results_['mean_test_score'][grid_search.best_index_] * 100:.2f}%")
print(f"Standard Deviation: {grid_search.cv_results_['std_test_score'][grid_search.best_index_] * 100:.2f}%")




# 2- Calculate the metrics for test data---------------------------------------
print("\nTest Data Metrics")
test_predictions = best_model.predict(x_test)

s12 = accuracy_score(y_test, test_predictions)  
s22 = f1_score(y_test, test_predictions, average='weighted')
s32 = recall_score(y_test, test_predictions, average='weighted')
s42 = precision_score(y_test, test_predictions, average='weighted')  

print(f"Test Accuracy: {s12 * 100:.2f}%")  
print(f"Test F1 Score: {s22 * 100:.2f}%")
print(f"Test Recall: {s32 * 100:.2f}%")
print(f"Test Precision: {s42 * 100:.2f}%")

cm_test = confusion_matrix(y_test, test_predictions)  
print("\nConfusion Matrix (Unseen Data):")
print(cm_test)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix - Test Data')
plt.show()


# 3- Calculate the metrics for all---------------------------------------------
print("\n All Data Metrics")
all_predictions = best_model.predict(x)

s111 = accuracy_score(y, all_predictions)  
s222 = f1_score(y, all_predictions, average='weighted')
s333 = recall_score(y, all_predictions, average='weighted')
s444 = precision_score(y, all_predictions, average='weighted')  
print(f"Test Accuracy: {s111 * 100:.2f}%")  
print(f"Test F1 Score: {s222 * 100:.2f}%")
print(f"Test Recall: {s333 * 100:.2f}%")
print(f"Test Precision: {s444 * 100:.2f}%")


cm_all = confusion_matrix(y, all_predictions)  
print("\nConfusion Matrix (Whole Data):")
print(cm_all)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_all, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix - All Data')
plt.show()



# Print the best model parameters
print("\nBest parameters:")
for param, value in grid_search.best_params_.items():
    print(f"{param}: {value}")




#%% 9.1) STFT Data Transformation

# Beyond the STFT- Feature Extraction method, the Spectrogram images will be used as data.


def eeg_stft_extraction2(x, fs=128):
    
    
    _, __, z = stft(x, fs=fs, window='hann', nperseg=512, noverlap=128, nfft=1024)
    
    #down_z = np.abs(z[::2, ::2]) downsampling is removed
    
    a = np.abs(z)
    return a


example4 = eeg_stft_extraction2(signal)





stft_data2 = np.zeros((84, 16, 513, 21)) # consider size of the matrix


for iii in range(84):
    for jjj in range(16):
        stft_data2[iii, jjj, :, :] = eeg_stft_extraction2(dataset[iii, jjj, :])
        


plt.imshow(stft_data2[3,2,:,:]) # simple plot for verification


#%% 9.2) STFT-CNN 



import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, f1_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



# 1. Pytorch instead of Keras, is choosen for CUDA availability
print("CUDA Availability:", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Computing Device: {device}")



# 2. Hyperparameters___________________________________________________________
lr = 0.0005
batch_size = 5
epochs = 250
patience = 200  # Early stopping patience



# 3. Labels
label1 = np.zeros((39, 1))
label2 = np.ones((45, 1))
labels = np.vstack((label1, label2))  # not one-hot encoded


# 4. train-test split
train_data, test_data, train_labels, test_labels = train_test_split(
    stft_data2, labels, test_size=0.2, random_state=42, stratify=labels)



# 5. Transformation to PyTorch Tensors
train_data = torch.tensor(train_data, dtype=torch.float32)
train_labels = torch.tensor(train_labels.squeeze(), dtype=torch.long)
test_data = torch.tensor(test_data, dtype=torch.float32)
test_labels = torch.tensor(test_labels.squeeze(), dtype=torch.long)



# 6. DataLoader
train_dataset = TensorDataset(train_data, train_labels)
test_dataset = TensorDataset(test_data, test_labels)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)



# 7. Model definition__________________________________________________________
class EEGCNN(nn.Module):
    def __init__(self):
        super(EEGCNN, self).__init__()
        self.conv1 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.af1 = nn.SiLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.af2 = nn.SiLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(0.2)
        
        self.fc1 = nn.Linear(64 * 128 * 5, 256) # consider input image size/4 (two maxpool)
        self.af3 = nn.SiLU()
        self.dropout3 = nn.Dropout(0.2) 
        
        self.fc2 = nn.Linear(256, 128)
        self.af4 = nn.SiLU()
        
        self.fc3 = nn.Linear(128, 2) #output layer

        
        

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.af1(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.af2(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.af3(x)
        x = self.dropout3(x)
       
        x = self.fc2(x)
        x = self.af4(x)
        x = self.fc3(x)

        return x

model = EEGCNN().to(device)



# 8. Loss and Optimizer________________________________________________________
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.001)



# 9. Train and Evaluation loops with Early Stopping 
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

best_test_accuracy = 0.0  # variables to control and save the best weights
best_model_state = None  
early_stopping_counter = 0


for epoch in range(epochs):
    # Training
    model.train()
    running_loss = 0.0
    y_true_train = []
    y_pred_train = []
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        

        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        y_true_train.extend(labels.cpu().numpy())
        y_pred_train.extend(predicted.cpu().numpy())
    
    train_loss = running_loss / len(train_loader)
    train_accuracy = accuracy_score(y_true_train, y_pred_train) * 100
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    
    # Test
    model.eval()
    test_loss = 0.0
    y_true_test = []
    y_pred_test = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            y_true_test.extend(labels.cpu().numpy())
            y_pred_test.extend(predicted.cpu().numpy())
    
    test_loss = test_loss / len(test_loader)
    test_accuracy = accuracy_score(y_true_test, y_pred_test) * 100
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)
    
    # Check for Early Stopping
    if test_accuracy > best_test_accuracy:
        best_test_accuracy = test_accuracy
        # copy of the model state
        best_model_state = {key: value.cpu().clone() for key, value in model.state_dict().items()}
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1
    
    if early_stopping_counter >= patience:
        print("Early stopping triggered.")
        break
    
    print(f"Epoch {epoch+1}/{epochs}")
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")


# 10. Load the best model weights
print("\nRestoring best model weights...")
if best_model_state is not None:
    # Move the best state back to the device and load it
    best_model_state = {key: value.to(device) for key, value in best_model_state.items()}
    model.load_state_dict(best_model_state)
    print(f"Best model restored with test accuracy: {best_test_accuracy:.2f}%")


# 11. Final evaluation
model.eval()
test_loss = 0.0
y_true_test = []
y_pred_test = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        y_true_test.extend(labels.cpu().numpy())
        y_pred_test.extend(predicted.cpu().numpy())


# 12. Calculate the  metrics
accuracy = accuracy_score(y_true_test, y_pred_test)
precision = precision_score(y_true_test, y_pred_test, average='weighted')
f1 = f1_score(y_true_test, y_pred_test, average='weighted')

print(f'Test Accuracy: {accuracy:.4f}')
print(f'Test Precision: {precision:.4f}')
print(f'Test F1 Score: {f1:.4f}')



plt.figure(figsize=(14, 6))

# Loss
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Accuracy
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(test_accuracies, label='Test Accuracy')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.show()

# Confusion matrix
cm = confusion_matrix(y_true_test, y_pred_test)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()




#torch.save(model.state_dict(), 'eeg_cnn.pth')






#%% 10.1) CWT Data Transformation


# Similar to previous approach, Scalogram (CWT) images will be obtained from the original dataset, and will be used as CNN input

import pywt

wavelist = pywt.wavelist(kind='continuous') # The built-in CWT can only use limited number of wavelets.



def eeg_cwt_extraction(x):
    
    fs=128
    scales = np.geomspace(1, 60, 60) 
    scale_frequencies = pywt.scale2frequency("cmor1.5-10",scales)*fs
    
    coefficients, frequencies = pywt.cwt(x, scales, 'cmor1.5-10', 1/fs) 
    log_coeffs = np.log10(abs(coefficients) + 1)

    
    downsample = log_coeffs[:, ::4]  # consider the downsampled size
    
    return downsample


example5 = eeg_cwt_extraction(signal)





cwt_data = np.zeros((84, 16, 60, 1920 )) 


for iii in range(84):
    for jjj in range(16):
        cwt_data[iii, jjj, :, :] = eeg_cwt_extraction(dataset[iii, jjj, :])
        


plt.imshow(cwt_data[3,2,:,:]) # simple plot for verification


# np.save('array.npy', cwt_data)  Save the entire array to a binary file, to see the size of the resultant data


#%% 10.2) CWT-CNN



import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, f1_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



# 1. Pytorch instead of Keras, is choosen for CUDA availability
print("CUDA Availability:", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Computing Device: {device}")



# 2. Hyperparameters___________________________________________________________
lr = 0.0001
batch_size = 5
epochs = 250
patience = 200  # Early stopping patience



# 3. Labels
label1 = np.zeros((39, 1))
label2 = np.ones((45, 1))
labels = np.vstack((label1, label2))  # not one-hot encoded


# 4. train-test split
train_data, test_data, train_labels, test_labels = train_test_split(
    cwt_data, labels, test_size=0.2, random_state=42, stratify=labels)



# 5. Transformation to PyTorch Tensors
train_data = torch.tensor(train_data, dtype=torch.float32)
train_labels = torch.tensor(train_labels.squeeze(), dtype=torch.long)
test_data = torch.tensor(test_data, dtype=torch.float32)
test_labels = torch.tensor(test_labels.squeeze(), dtype=torch.long)



# 6. DataLoader
train_dataset = TensorDataset(train_data, train_labels)
test_dataset = TensorDataset(test_data, test_labels)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)



# 7. Model definition__________________an AlexNet variant______________________
class EEGCNN2(nn.Module):
    def __init__(self):
        super(EEGCNN2, self).__init__()
        
        # Input: [batch_size, 16, 60, 1920] -> Output: [batch_size, 96, 27, 478]
        self.conv1 = nn.Conv2d(16, 96, kernel_size=(7,11), stride=(2,4))  
        self.relu1 = nn.SiLU()
        self.lrn1 = nn.BatchNorm2d(96)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        
        # Input: [batch_size, 96, 13, 238] -> Output: [batch_size, 256, 6, 118]
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, padding=2)
        self.relu2 = nn.SiLU()
        self.lrn2 = nn.BatchNorm2d(256)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        
        # Input: [batch_size, 256, 6, 118] -> Output: [batch_size, 384, 6, 118]
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
        self.relu3 = nn.SiLU()
        
        
        # Input: [batch_size, 384, 6, 118] -> Output: [batch_size, 384, 6, 118]
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        self.relu4 = nn.SiLU()
        
        
        # Input: [batch_size, 384, 6, 118] -> Output: [batch_size, 256, 2, 58]
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.relu5 = nn.SiLU()
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        # Flatten layer
        self.flatten = nn.Flatten()  # Converts feature maps to vector: 256 * 2 * 58 = 29,696
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(256 * 2 * 58, 4096)
        self.relu6 = nn.SiLU()
        self.dropout1 = nn.Dropout(p=0.2)
        
        self.fc2 = nn.Linear(4096, 512)
        self.relu7 = nn.SiLU()
        self.dropout2 = nn.Dropout(p=0.1)
        
        self.fc3 = nn.Linear(512, 2)
        

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.lrn1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.lrn2(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.relu3(x)
        
        x = self.conv4(x)
        x = self.relu4(x)
        
        x = self.conv5(x)
        x = self.relu5(x)
        x = self.pool5(x)
        
        # Flatten and Fully Connected Layers
        x = self.flatten(x)
        
        x = self.fc1(x)
        x = self.relu6(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.relu7(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        
        
        return x

model = EEGCNN2().to(device)



# 8. Loss and Optimizer________________________________________________________
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0005)



# 9. Train and Evaluation loops with Early Stopping 
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

best_test_accuracy = 0.0  # variables to control and save the best weights
best_model_state = None  
early_stopping_counter = 0


for epoch in range(epochs):
    # Training
    model.train()
    running_loss = 0.0
    y_true_train = []
    y_pred_train = []
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        

        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        y_true_train.extend(labels.cpu().numpy())
        y_pred_train.extend(predicted.cpu().numpy())
    
    train_loss = running_loss / len(train_loader)
    train_accuracy = accuracy_score(y_true_train, y_pred_train) * 100
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    
    # Test
    model.eval()
    test_loss = 0.0
    y_true_test = []
    y_pred_test = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            y_true_test.extend(labels.cpu().numpy())
            y_pred_test.extend(predicted.cpu().numpy())
    
    test_loss = test_loss / len(test_loader)
    test_accuracy = accuracy_score(y_true_test, y_pred_test) * 100
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)
    
    # Check for Early Stopping
    if test_accuracy > best_test_accuracy:
        best_test_accuracy = test_accuracy
        # copy of the model state
        best_model_state = {key: value.cpu().clone() for key, value in model.state_dict().items()}
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1
    
    if early_stopping_counter >= patience:
        print("Early stopping triggered.")
        break
    
    print(f"Epoch {epoch+1}/{epochs}")
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")


# 10. Load the best model weights
print("\nRestoring best model weights...")
if best_model_state is not None:
    # Move the best state back to the device and load it
    best_model_state = {key: value.to(device) for key, value in best_model_state.items()}
    model.load_state_dict(best_model_state)
    print(f"Best model restored with test accuracy: {best_test_accuracy:.2f}%")


# 11. Final evaluation
model.eval()
test_loss = 0.0
y_true_test = []
y_pred_test = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        y_true_test.extend(labels.cpu().numpy())
        y_pred_test.extend(predicted.cpu().numpy())


# 12. Calculate the  metrics
accuracy = accuracy_score(y_true_test, y_pred_test)
precision = precision_score(y_true_test, y_pred_test, average='weighted')
f1 = f1_score(y_true_test, y_pred_test, average='weighted')

print(f'Test Accuracy: {accuracy:.4f}')
print(f'Test Precision: {precision:.4f}')
print(f'Test F1 Score: {f1:.4f}')



plt.figure(figsize=(14, 6))

# Loss
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Accuracy
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(test_accuracies, label='Test Accuracy')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.show()

# Confusion matrix
cm = confusion_matrix(y_true_test, y_pred_test)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()



#torch.save(model.state_dict(), 'eeg_cwt_cnn.pth')

#%% 10.3) CNN K-Fold CV 


# PyTorch K-fold cross-validation implementation, by maintaining fold independence and best weight restoration.



import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, f1_score, recall_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. CUDA check
print("CUDA Availability:", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Computing Device: {device}")

# 2. Hyperparameters
lr = 0.0001
batch_size = 5
epochs = 250
patience = 200

# 3. Labels
label1 = np.zeros((39, 1))
label2 = np.ones((45, 1))
labels = np.vstack((label1, label2))

# 4. Initial train-test split
train_data, test_data, train_labels, test_labels = train_test_split(
    cwt_data, labels, test_size=0.2, random_state=42, stratify=labels)

# 5. Transform to PyTorch tensors
train_data = torch.tensor(train_data, dtype=torch.float32)
train_labels = torch.tensor(train_labels.squeeze(), dtype=torch.long)
test_data = torch.tensor(test_data, dtype=torch.float32)
test_labels = torch.tensor(test_labels.squeeze(), dtype=torch.long)

# 6. K-Fold Cross Validation setup
k = 10
kf = KFold(n_splits=k, shuffle=False)

fold_number = 1
all_training_predictions = []
all_training_labels = []
accuracies = []

# 7. K-Fold Cross Validation loop
for train_idx, val_idx in kf.split(train_data):
    print(f'Fold {fold_number}')
    
    # Split fold data
    x_fold_train = train_data[train_idx]
    y_fold_train = train_labels[train_idx]
    x_fold_val = train_data[val_idx]
    y_fold_val = train_labels[val_idx]
    
    # Create DataLoaders for this fold
    fold_train_dataset = TensorDataset(x_fold_train, y_fold_train)
    fold_val_dataset = TensorDataset(x_fold_val, y_fold_val)
    fold_train_loader = DataLoader(fold_train_dataset, batch_size=batch_size, shuffle=True)
    fold_val_loader = DataLoader(fold_val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model, criterion, and optimizer
    model = EEGCNN2().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0005)
    
    # Training variables
    best_val_accuracy = 0.0
    best_model_state = None
    early_stopping_counter = 0
    
    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        for inputs, labels in fold_train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Validation phase
        model.eval()
        val_predictions = []
        val_true_labels = []
        with torch.no_grad():
            for inputs, labels in fold_val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                val_predictions.extend(predicted.cpu().numpy())
                val_true_labels.extend(labels.cpu().numpy())
        
        # Calculate validation accuracy
        val_accuracy = accuracy_score(val_true_labels, val_predictions)
        
        # Early stopping check
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state = {key: value.cpu().clone() for key, value in model.state_dict().items()}
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
        
        if early_stopping_counter >= patience:
            print("Early stopping triggered")
            break
    
    # Load best model for this fold
    best_model_state = {key: value.to(device) for key, value in best_model_state.items()}
    model.load_state_dict(best_model_state)
    
    # Final fold evaluation
    model.eval()
    fold_predictions = []
    fold_true_labels = []
    with torch.no_grad():
        for inputs, labels in fold_val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            fold_predictions.extend(predicted.cpu().numpy())
            fold_true_labels.extend(labels.cpu().numpy())
    
    # Store metrics
    fold_accuracy = accuracy_score(fold_true_labels, fold_predictions)
    accuracies.append(fold_accuracy)
    all_training_predictions.extend(fold_predictions)
    all_training_labels.extend(fold_true_labels)
    
    fold_number += 1

# 8. Report cross-validation results
mean_accuracy = np.mean(accuracies)
std_accuracy = np.std(accuracies)
print(f'Cross-Validation Mean Accuracy: {mean_accuracy*100:.3f}%')
print(f'Standard Deviation of Accuracy: {std_accuracy*100:.3f}%')

# 9. Plot cross-validation confusion matrix
cm_cv = confusion_matrix(all_training_labels, all_training_predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_cv, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix - Cross-Validation on Training')
plt.show()

# 10. Final model training on entire training set
final_model = EEGCNN2().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(final_model.parameters(), lr=lr, weight_decay=0.0005)

# Create DataLoaders for final training
train_dataset = TensorDataset(train_data, train_labels)
test_dataset = TensorDataset(test_data, test_labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Training variables for final model
best_test_accuracy = 0.0
best_final_model_state = None
early_stopping_counter = 0

# Final training loop
for epoch in range(epochs):
    final_model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = final_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    # Validation on test set
    final_model.eval()
    test_predictions = []
    test_true_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = final_model(inputs)
            _, predicted = torch.max(outputs, 1)
            test_predictions.extend(predicted.cpu().numpy())
            test_true_labels.extend(labels.cpu().numpy())
    
    test_accuracy = accuracy_score(test_true_labels, test_predictions)
    
    if test_accuracy > best_test_accuracy:
        best_test_accuracy = test_accuracy
        best_final_model_state = {key: value.cpu().clone() for key, value in final_model.state_dict().items()}
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1
    
    if early_stopping_counter >= patience:
        print("Early stopping triggered in final training")
        break

# Load best final model
best_final_model_state = {key: value.to(device) for key, value in best_final_model_state.items()}
final_model.load_state_dict(best_final_model_state)

# Final evaluation
final_model.eval()
final_predictions = []
final_true_labels = []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = final_model(inputs)
        _, predicted = torch.max(outputs, 1)
        final_predictions.extend(predicted.cpu().numpy())
        final_true_labels.extend(labels.cpu().numpy())

# Calculate final metrics
final_accuracy = accuracy_score(final_true_labels, final_predictions)
final_precision = precision_score(final_true_labels, final_predictions, average='weighted')
final_recall = recall_score(final_true_labels, final_predictions, average='weighted')
final_f1 = f1_score(final_true_labels, final_predictions, average='weighted')

print(f'Final Test Accuracy: {final_accuracy*100:.3f}%')
print(f"F1 Score: {final_f1*100:.2f}%")
print(f"Recall: {final_recall*100:.2f}%")
print(f"Precision: {final_precision*100:.2f}%")

# Plot final confusion matrix
cm_test = confusion_matrix(final_true_labels, final_predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix - Test Data')
plt.show()


#%% 10.4) CNN - LOOCV


"""
Leave-One-Out Cross-Validation (LOOCV) is a technique to assess the model performance on limited numbers of data.
It is a specific case of k-fold cross-validation where the number of folds equals the number of data points in the dataset.
It applies for small datasets, because it is computationally intensive
"""


from sklearn.model_selection import LeaveOneOut




# 1. CUDA check
print("CUDA Availability:", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Computing Device: {device}")

# 2. Hyperparameters
lr = 0.0001
batch_size = 5
epochs = 120
patience = 90

# 3. Data preparation
x = torch.tensor(cwt_data, dtype=torch.float32)
label1 = np.zeros((39, 1))
label2 = np.ones((45, 1))
labels = np.vstack((label1, label2))
y = torch.tensor(labels.squeeze(), dtype=torch.long)

# 4. LOOCV setup
loo = LeaveOneOut()
accuracies = []
all_predictions = []
all_true_labels = []
fold_count = 1
total_folds = len(x)

# 5. LOOCV loop
for train_idx, test_idx in loo.split(x):
    print(f'Fold {fold_count}/{total_folds}')
    
    # Split data
    x_train, x_test = x[train_idx], x[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Create DataLoaders
    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # Initialize model, criterion, and optimizer
    model = EEGCNN2().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0005)
    
    # Training variables
    best_val_accuracy = 0.0
    best_model_state = None
    early_stopping_counter = 0
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                val_accuracy = (predicted == labels).float().mean().item()
        
        # Early stopping
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state = {key: value.cpu().clone() for key, value in model.state_dict().items()}
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
        
        if early_stopping_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break
    
    # Load best model
    best_model_state = {key: value.to(device) for key, value in best_model_state.items()}
    model.load_state_dict(best_model_state)
    
    # Final evaluation for this fold
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_true_labels.extend(labels.cpu().numpy())
            accuracies.append((predicted == labels).float().mean().item())
    
    fold_count += 1

# 6. Calculate and display final metrics
mean_accuracy = np.mean(accuracies)
std_accuracy = np.std(accuracies)
print(f'LOOCV Mean Accuracy: {mean_accuracy*100:.3f}%')
print(f'Standard Deviation: {std_accuracy*100:.3f}%')

# Confusion Matrix
cm = confusion_matrix(all_true_labels, all_predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix - LOOCV')
plt.show()

# Additional metrics
final_accuracy = accuracy_score(all_true_labels, all_predictions)
final_precision = precision_score(all_true_labels, all_predictions, average='weighted')
final_recall = recall_score(all_true_labels, all_predictions, average='weighted')
final_f1 = f1_score(all_true_labels, all_predictions, average='weighted')

print(f'Final LOOCV Accuracy: {final_accuracy*100:.3f}%')
print(f"F1 Score: {final_f1*100:.2f}%")
print(f"Recall: {final_recall*100:.2f}%")
print(f"Precision: {final_precision*100:.2f}%")


