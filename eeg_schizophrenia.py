
"""

_________EEG SCHIZOPHRENIA CLASSIFICATION_________

Oguzhan Memis  January 2025



DATASET DESCRIPTION:
__________________________________________________
    
EEG Dataset which contains 2 classes of EEG signals captured from adolescents.

-Classes: Normal (39 people) and Schizophrenia (45 people).

-Properties:
    
    16 channels * 128 sample-per-second * 60 seconds of measurement for each person.
    
    Voltages are captured in units of microvolts (µV) 10^-6



-Orientation:
    
    Signals are vertically placed into text files, ordered by channel number (1 to 16).
    
    Length of 1 signal is = 128*60 = 7680 samples.

    So each text file contains  16*7680 = 122880 samples , vertically.


SOURCE:

http://brain.bio.msu.ru/eeg_schizophrenia.htm 

Original article of dataset:  https://doi.org/10.1007/s10747-005-0042-z  Physiology (Q4) 

A recent article which uses this dataset: https://doi.org/10.1007/s11571-024-10121-0  Cognitive Neuroscience (Q2)




THE CODE IS DIVIDED INTO SEPARATE CELLS, RUN EACH CELL ONE BY ONE CONSECUTIVELY

"""




#%% 1) Importing the data

import os
import numpy as np
import matplotlib.pyplot as plt



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
scale_frequencies = pywt.scale2frequency("morl",scales)*fs  # always verify it to represent accurate frequency resolution

# Difference than spectrogram, comes from these uneven distribution of wavelet frequencies.
coefficients, frequencies = pywt.cwt(signal, scales, 'morl', 1/fs) 
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
scales = np.geomspace(1, 60, 60)  
scale_frequencies = pywt.scale2frequency("morl",scales)*fs  

coefficients, frequencies = pywt.cwt(signal, scales, 'morl', 1/fs) 
log_coeffs = np.log10(abs(coefficients) + 1)  # prevent log(0)


plt.figure(figsize=(13, 8))
f_min = 0
f_max = fs/2
plt.imshow(log_coeffs, extent=[0, len(signal)/fs, f_min, f_max], 
           aspect='auto', cmap='viridis', interpolation='bilinear' ) 

cbar = plt.colorbar()
cbar.set_label('Log10(Magnitude + 1)')


y_axis_labels =   list(map(int, np.flip(frequencies)))             
step = 10                                                
plt.yticks(
    ticks=np.arange(0, len(y_axis_labels), step),  
    labels=y_axis_labels[::step]                         
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




#%% 5) Connectivity maps and plots

import seaborn as sns
from scipy.spatial.distance import pdist, squareform



# 1) Correlation matrix between the channels, for every individual

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




A5, D5, D4, D3, D2, D1 = pywt.wavedec(signal2, 'bior3.3', level=5)  # Perform wavelet decomposition (db4 wavelet, 5 levels)


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


# Calculate the metrics for all---------------------------------------------
print("\n All Data Metrics")
all_predictions = best_model.predict(x)

s111 = accuracy_score(y, all_predictions)  
s222 = f1_score(y, all_predictions, average='weighted')
s333 = recall_score(y, all_predictions, average='weighted')
s444 = precision_score(y, all_predictions, average='weighted')  
print(f"All Accuracy: {s111 * 100:.2f}%")  
print(f"All F1 Score: {s222 * 100:.2f}%")
print(f"All Recall: {s333 * 100:.2f}%")
print(f"All Precision: {s444 * 100:.2f}%")


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



#%% 7.4) MLP model iterations


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

x_train, x_test, y_train, y_test = train_test_split(x, onehot,test_size=0.3 , random_state=42, stratify=onehot)




optimizer = SGD(learning_rate=0.0008, momentum=0.4)

early_stop= EarlyStopping(monitor='val_accuracy', patience=150, restore_best_weights=True)

act = "silu"

# Model 
model6 = Sequential()
model6.add(Dense(200, activation=act, input_dim=400)) 
model6.add(Dropout(0.3))
model6.add(Dense(200, activation=act, kernel_regularizer=l2(0.003)))
model6.add(Dropout(0.2))
model6.add(Dense(50, activation=act, kernel_regularizer=l2(0.002)))
model6.add(Dense(2,activation="softmax")) 
model6.compile(optimizer=optimizer,loss="binary_crossentropy", metrics=["accuracy"])  
history=model6.fit(x_train, y_train, validation_data=(x_test,y_test),epochs=220,batch_size=5, callbacks=[early_stop])

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




#%%  Optional: Save and reuse the best model

#model6.save('mlp_model.h5')


from keras.models import load_model

model = load_model('mlp_model.h5')


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



#%% 7.5) MLP K-Fold CV 

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
x_train, x_test, y_train, y_test = train_test_split(x, onehot, test_size=0.3, random_state=42, stratify=onehot)



# 2-Custom function to generate the model where it is needed
def make_model():
    model = Sequential()
    model.add(Dense(150, activation='silu', input_dim=400))
    model.add(Dropout(0.3))
    model.add(Dense(150, activation="silu", kernel_regularizer=l2(0.002)))
    model.add(Dropout(0.2))
    model.add(Dense(50, activation="silu", kernel_regularizer=l2(0.002)))
    model.add(Dense(2, activation="softmax"))
    optimizer = RMSprop(learning_rate=0.0005, rho=0.8, epsilon=1e-08)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    return model

# 3-Definitions for K-Fold Cross-Validation

k = 5  # Number of folds
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

    early_stop = EarlyStopping(monitor='val_accuracy', patience=150, restore_best_weights=True)

    
    model = make_model()
    history = model.fit(x_fold_train, y_fold_train, validation_data=(x_fold_val, y_fold_val), epochs=200, batch_size=5, callbacks=[early_stop], verbose=0)

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
plt.title('Confusion Matrix - Cross-Validation')
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








#%% 7.6) Leave One Out CV 





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

    optimizer = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.9)  # Learning Rate
    early_stop = EarlyStopping(monitor='val_accuracy', patience=100, restore_best_weights=True)

    # Model
    model = Sequential()
    model.add(Dense(90, activation='silu', input_dim=400))
    model.add(Dropout(0.2))
    model.add(Dense(90, activation="silu", kernel_regularizer=l2(0.005)))
    model.add(Dropout(0.2))
    model.add(Dense(45, activation="silu", kernel_regularizer=l2(0.005)))
    model.add(Dense(2, activation="softmax"))
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=120, batch_size=5, callbacks=[early_stop], verbose=0)

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

# In this method, the Spectrogram of the signal (result of STFT) will be given into MLP input


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
    
    down_z = np.abs(z[::4, ::4]) # downsampling by factor of 4
    
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


#%% 8.2) MLP



import keras 
from keras.models import Sequential
from keras.layers import Dense,Dropout ,BatchNormalization
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

x_train, x_test, y_train, y_test = train_test_split(x2, onehot2,test_size=0.3 , random_state=42, stratify=onehot2)


# Standard Normalization
scale = StandardScaler()
scale.fit(x_train) # Adapting to just training set
x_train = scale.transform(x_train) # Transform
x_test = scale.transform(x_test) 


myactivation = "silu"
early_stop= EarlyStopping(monitor='val_accuracy', patience=150, restore_best_weights=True)
optimizer = Adam(learning_rate=0.01)


# Model 
model6 = Sequential()
model6.add(Dense(150, activation= myactivation, input_dim=5200))
model6.add(Dropout(0.5))
model6.add(Dense(30, activation=myactivation, kernel_regularizer=l2(0.05)))
model6.add(Dropout(0.5))
model6.add(Dense(10, activation=myactivation, kernel_regularizer=l2(0.01)))
model6.add(Dense(2,activation="softmax")) 
model6.compile(optimizer=optimizer ,loss="categorical_crossentropy", metrics=["accuracy"])  
history=model6.fit(x_train, y_train, validation_data=(x_test,y_test),epochs=200,batch_size=5, callbacks=[early_stop])

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


#%% 8.3) SVM 




from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale , StandardScaler
from sklearn.metrics import accuracy_score , f1_score , recall_score, precision_score , confusion_matrix
from sklearn.model_selection import GridSearchCV



# Matrices are flattened to a 1D vector 
x = stft_data1.reshape(84, -1)  


y = np.array([0] * 39 + [1] * 45)  # Binary labels



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42, stratify=y)




mymodel = svm.SVC(probability=True) # Classifier model



# Define the parameter grid
param_grid = {
    'C': [0.01,0.1,1,2,3,4,5,6,7,8,9,10,15,20,25,30,40,50,100,150,200,300,500],
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


  


#%% Extra methods: pretrained FE and CNNs


