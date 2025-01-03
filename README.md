# EEG_Schizophrenia
 Binary classification (Schizophrenia / Normal) of the adolescent EEG signals, obtained from an open-access dataset.
 <br>
 In this project, various Time and Frequency feature extraction methods (DWT, STFT and CWT) are applied to the EEG signals, in order to obtain better classification performance. The result of %94 test accuracy is obtained by the DWT-MLP method which uses spectral features, and by the CWT-CNN method. 

 <br>
 Oguzhan Memis  January 2025

<br><br>

## Content
1) Files in this repository
2) Dataset desription
3) Code Organization
4) Considerations
5) Reference

<br><br><br>

## 1-Files in this repository

Code file is **"eeg_schizophrenia.py"** and dataset file is **"dataset_text.zip"** which includes two folders.
There is also a model file called **"dwt_mlp_model_96.h5"** to import and use for the DWT method. Relevant instructions are noted in later sections.

<br><br><br>

## 2-Dataset Description

EEG Dataset which contains 2 classes of EEG signals captured from adolescents.
<br>
-Classes: Normal (39 people) and Schizophrenia (45 people).
<br>
-Properties:
<br>   
    *16 channels * 128 sample-per-second * 60 seconds of measurement for each person.
<br>  
    *Voltages are captured in units of microvolts (µV) 10^-6
<br>
    *So the amplitudes of the signals varies from -2000 to +2000
<br><br>

-Orientation:
<br>    
    *Signals are vertically placed into text files, ordered by channel number (1 to 16).
<br>    
    *The length of 1 signal is = 128*60 = 7680 samples.
<br>
    *So each text file contains  16*7680 = 122880 samples, vertically.

<br><br>
Source of the dataset: [Moscow State University 2005](http://brain.bio.msu.ru/eeg_schizophrenia.htm) 
<br>
The original article of the dataset:  [2005 Borisov et al.](https://doi.org/10.1007/s10747-005-0042-z)  Physiology (Q4) 
<br>
A recent article that uses this dataset: [2024 Bagherzadeh & Shalbaf](https://doi.org/10.1007/s11571-024-10121-0)  Cognitive Neuroscience (Q2)

<br><br><br>



## 3-Code Organization:


The codes are divided into separate cells by putting #%%,
<br>
RUN EACH CELL ONE BY ONE CONSECUTIVELY.
    <br>
    
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

<br><br><br>
    
## 4-Considerations:

*Before running the classification models, consider related data transformation/feature extraction methods
 and the input size (for the Deep Learning models). 
<br>
*The DWT-Feature extraction method gives an output dataset in size of (84,16,25)
 then the data of every subject are flattened into 16*25=400
 <br>
*Use different wavelets for SVM and the MLP models. Such as 'bior2.8' and 'bior3.3' for the SVM
 <br>
*The first STFT-Feature extraction method gives an output dataset in size of (84,16,325)
 It uses a downsampled and flattened STFT.
 Then the data of every subject are flattened into 16*325=5200
<br>
*In the second STFT method, Spectrograms of the signals are not flattened, and 
 dataset in size of  (84, 16, 513, 21) is obtained. 
 The CNN model takes the input as 16 channel 513*21 matrices.
<br>
*In the last CWT method, Scalograms (downsampled in one axis) of the signals are captured 
 into the resultant dataset which has a size of (84, 16, 60, 1920).
 The CNN model takes the input as 16 channel 60*1920 matrices.
<br>
*All the MLP models are built by using Keras, 
 and all the CNN models are built by using PyTorch (uses GPU) 

<br><br><br>

## 5-Reference this repository

Please refer with the name of the repository owner Oğuzhan Memiş, with the link of this repository.
Also don't forget to cite the dataset owner [2005 Borisov et al.](https://doi.org/10.1007/s10747-005-0042-z).

<br>

Contacts and suggestions are welcomed: memisoguzhants@gmail.com 