# urban-sound-classification


[![Dataset](https://img.shields.io/badge/Dataset-Urban8k-red.svg)](https://www.kaggle.com/pavansanagapati/urban-sound-classification/downloads/urban-sound-classification.zip/2)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/)
![stability-experimental](https://img.shields.io/badge/stability-experimental-orange.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)

Urban sound source tagging from an aggregation of four second noisy audio clips via 1D and 2D CNN (Xception)

## Dataset Description
The Urban Sound Classification dataset contains 8732 labeled sound excerpts (<=4s) of urban sounds from 10 classes,namely: 

* Air Conditioner 
* Car Horn 
* Children Playing
* Dog bark 
* Drilling Engine 
* Idling Gun Shot
* Jackhammer
* Siren 
* Street Music 

The attributes of data are mapped as follows: 
* **ID** – Unique ID of sound excerpt and **Class** – type of sound

![air_conditioner](https://github.com/rednafi/urban-sound-classification/blob/master/notebooks/eda_plots/amplitude_vs_time/air_conditioner.svg)
![air_conditioner](https://user-images.githubusercontent.com/30027932/57352070-febe8a80-7185-11e9-8806-44ccfb79d986.png)

## Project Organization
### Folder Structure
```
.
├── data/
   ├── train/
      ├── Train/
         ├── 1.wav
         ├── 2.wav
         .............
         
   ├── test/
      ├── Test/
         ├── 5.wav
         ├── 7.wav
         .............
         
    ├── train.csv
    ├── test.csv
            
├── models/
   ├── model_1d.h5
   ├── model_2d.h5
    
├── notebooks/
    ├── eda_plots/
       ├── amplitude_vs_time/
          ├── air_conditioner.svg
          ├── car_horn.svg
          ............. 
    
       ├── mel_spectrum/
          ├── air_conditioner.svg
          ├── car_horn.svg
          .............      
     
    ├── Exploratory Data Analysis.ipynb

├── results/
    ├── pred_1d.csv
    ├── pred_2d.csv

├── src/
    ├── utils_1d.py
    ├── utils_2d.py
    ├── train_1d.py
    ├── train_2d.py
    ├── test_1d.py
    ├── test_2d.py


├── LICENSE
├── requirements.txt
    
```

### Workflow 

**Exploratory Data Analysis:** 
* Frequency normalization and plotting amplitude vs time
* Mel spectogram plotting 

**Audio Tagging:**

* Normalizing the audio clips and passing them through stacks of 1D convolution layers for feature extraction. Then the usual dense layer stacks were used to do the final categorization.

* Extracting features in the form of [mel-spectogram](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum) and passing them 
through stacks of 2D convolution layers for additional feature pulling. Dense layer stack does the final classification. In this case, we trained an Xception model from scratch to achieve better generalization capability.

### Result 

We achieved  89% validation accuracy in the second approach.
![xception_val_acc](https://github.com/rednafi/urban-sound-classification/blob/master/results/acc_model_2d.png)

### Requirements
```
pip install -r requirements.txt
```
