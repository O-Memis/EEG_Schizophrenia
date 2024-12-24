
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




A5, D5, D4, D3, D2, D1 = pywt.wavedec(signal2, 'db4', level=5)  # Perform wavelet decomposition (db4 wavelet, 5 levels)


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





#%% 7) Feature Extraction and Data Transformation

"""
After all the pre-processing and the observations, now the data should be prepared for the model input.

Many Feature Extraction, Data Transformation or Dimensionality Reduction methods can be applied such as:
Multi-level Decomposition by DWT, Time and Frequency statistics, EMD, PCA, Spectrogram, Recurrence Plot or similar. 
"""

from scipy import stats, integrate


dataset = np.concatenate((filtered_normal, filtered_patient))




#1) Extract statistical features from DWT decomposition coefficients

def extract_dwt_features(signal):
   
    dwt = pywt.wavedec(signal, 'bior5.5', level=5)
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






#%% 8) Model training




from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale , StandardScaler
from sklearn.metrics import accuracy_score , f1_score , recall_score, precision_score , confusion_matrix
from sklearn.model_selection import GridSearchCV



# Matrices are flattened to a 1D vector 
x = dwt_data.reshape(84, -1)  


y = np.array([0] * 39 + [1] * 45)  # Binary labels


x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.3 , random_state=42)





mymodel = svm.SVC(probability=True) # Classifier model

mymodel.get_params()  # Look for which parameters can be choosen




'''
Grid search is about simply iterating all sets of values defined, to obtain best performed model.
It also uses Cross-Validation. And parameter grid need to be defined.
'''


# Define the parameter grid
param_grid = {
    'C': [8.35, 8.4, 8.445, 8.45, 8.455, 8.5, 8.55, 8.6, 8.65, 8.68, 8.7, 8.72, 8.75, 8.8, 8.9, 8.95, 9, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.65, 9.7, 9.73, 9.75, 9.77],
    'kernel': ['linear', 'rbf', 'sigmoid'],
    'gamma': ['scale', 'auto'],
    'coef0': [0.0, 0.1, 0.5, 1.0],  # for polynomial and sigmoid kernels
    'shrinking': [True, False],

}


grid_search = GridSearchCV(mymodel, param_grid, cv=3, n_jobs=-1, verbose=0, scoring='accuracy')
# Setting n_jobs to -1 means that the algorithm will use all available CPU cores on your machine
# Verbose setting is for observing the details of the ongoing process 

grid_search.fit(x_train, y_train)



# Getting the best model
best_model = grid_search.best_estimator_




# Calculate the metrics for classification 

predictions = best_model.predict(x_test) # Make predictions

s1 = accuracy_score(y_test, predictions)
s2 = f1_score(y_test, predictions, average='weighted')
s3 = recall_score(y_test, predictions, average='weighted')
s4 = precision_score(y_test, predictions)

print(f"Accuracy: {s1*100:.2f}%")
print(f"F1 Score: {s2*100:.2f}%")
print(f"Recall: {s3*100:.2f}%")
print(f"Precision: {s4*100:.2f}%")


# Confusion matrix
cm = confusion_matrix(y_test, predictions)
print("Confusion Matrix:")
print(cm)


plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')  # Heatmap of confusion matrix
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix')
plt.show()


# Print the best model parameters
print("Best parameters:")
for param, value in grid_search.best_params_.items():
    print(f"{param}: {value}")




#%% 9) MLP model


import keras 
from keras.models import Sequential
from keras.layers import Dense,Dropout 
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

x_train, x_test, y_train, y_test = train_test_split(x, onehot,test_size=0.3 , random_state=42)




optimizer = Adam(learning_rate=0.0001, beta_1=0.95, beta_2=0.9) # Learning Rate

early_stop= EarlyStopping(monitor='val_accuracy', patience=25, restore_best_weights=True)


# Model 
model6 = Sequential()
model6.add(Dense(90, activation='silu', input_dim=400))  # Initializer
model6.add(Dropout(0.2))
model6.add(Dense(90,activation="silu", kernel_regularizer=l2(0.005)))
model6.add(Dropout(0.2))
model6.add(Dense(45,activation="silu", kernel_regularizer=l2(0.005)))
model6.add(Dense(2,activation="softmax")) 
model6.compile(optimizer=optimizer,loss="binary_crossentropy", metrics=["accuracy"])  
history=model6.fit(x_train, y_train, validation_data=(x_test,y_test),epochs=120,batch_size=5, callbacks=[early_stop])

predictions= model6.predict(x_test)  
model6.summary()                           





# The confusion matrix
Predictions=np.argmax(predictions, axis=1)
Y_test=np.argmax(y_test, axis=1)
cm = confusion_matrix(Y_test, Predictions)
print(cm)    
    

fig6=plt.figure()
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues') 
plt.xlabel('Predicted label')
plt.ylabel('True label') 
plt.title('Confusion  matrix') 
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




#%% Save and reuse the best model

#model6.save('mlp_model.h5')


from keras.models import load_model

model = load_model('mlp_model.h5')


newpredictions = model.predict(x_test) # Make predictions


newpredictions = np.argmax(newpredictions, axis=1)




s1 = accuracy_score(Y_test, newpredictions)
s2 = f1_score(Y_test, newpredictions, average='weighted')
s3 = recall_score(Y_test, newpredictions, average='weighted')
s4 = precision_score(Y_test, newpredictions)

print(f"Accuracy: {s1*100:.2f}%")
print(f"F1 Score: {s2*100:.2f}%")
print(f"Recall: {s3*100:.2f}%")
print(f"Precision: {s4*100:.2f}%")


# Confusion matrix
cm = confusion_matrix(Y_test, newpredictions)
print("Confusion Matrix:")
print(cm)


plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')  # Heatmap of confusion matrix
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix')
plt.show()


#%% 10) Extra methods: Image classification by CWT and STFT



