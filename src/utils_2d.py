# basic library imports
import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import random
from tqdm import tqdm

# pandas setting 
pd.set_option('display.max_columns', None)  
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)
pd.options.display.max_rows = 5000

# encoding the classes
from sklearn.preprocessing import LabelEncoder

# librosa for audion feature extraction
import librosa
import gc
import pickle
import random
from multiprocessing import Pool
from PIL import Image
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

input_length = 16000*4
batch_size = 16
n_mels = 320

def input_to_target(base_data_path = './data/'):

    # audio files and their corresponding labels
    train_path = base_data_path + "train/Train/*.wav"
    train_label_path = base_data_path +  "train.csv"
    test_path =  base_data_path + "test/Test/*.wav"

    # input
    train_files = glob.glob(train_path)
    train_files = pd.DataFrame({'train_file_paths': train_files})
    train_files['ID'] = train_files['train_file_paths'].apply(lambda x:x.split('/')[-1].split('.')[0])
    train_files['ID'] = train_files['ID'].astype(int)
    train_files = train_files.sort_values(by='ID')
    test_files = glob.glob(test_path)

    # target
    train_labels = pd.read_csv(train_label_path)
    train_file_to_label = train_files.merge(train_labels, on= "ID", how='inner')

    # encoding the classes
    int_encode = LabelEncoder()
    train_file_to_label['class_int_encode'] = int_encode.fit_transform(train_file_to_label['Class'])
    
    
    return train_file_to_label


def audio_normalization(data):
    
    max_data = np.max(data)
    min_data = np.min(data)
    data = (data-min_data)/(max_data-min_data+0.0001)
    return data-0.5

input_length = 16000*4

batch_size = 32
n_mels = 320

def mel_spectrum_db(audio, sample_rate=16000, window_size=20, #log_specgram
                 step_size=10, eps=1e-10):

    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels= n_mels)
    mel_db = (librosa.power_to_db(mel_spec, ref=np.max) + 40)/40

    return mel_db.T


def stretch(data, rate=1):

    data = librosa.effects.time_stretch(data, rate)
    if len(data)>input_length:
        data = data[:input_length]
    else:
        data = np.pad(data, (0, max(0, input_length - len(data))), "constant")

    return data

def pitch_shift(data, n_steps=3.0):

    data = librosa.effects.pitch_shift(data, sr=input_length, n_steps=n_steps)
    if len(data)>input_length:
        data = data[:input_length]
    else:
        data = np.pad(data, (0, max(0, input_length - len(data))), "constant")

    return data


def loguniform(low=0.00000001, high=0.01):
    return np.exp(np.random.uniform(np.log(low), np.log(high)))


def white(N, state=None):
    """
    White noise.
    :param N: Amount of samples.
    :param state: State of PRNG.
    :type state: :class:`np.random.RandomState`
    White noise has a constant power density. It's narrowband spectrum is therefore flat.
    The power in white noise will increase by a factor of two for each octave band,
    and therefore increases with 3 dB per octave.
    """
    state = np.random.RandomState() if state is None else state
    return state.randn(N)

def augment(data):
    if np.random.uniform(0, 1)>0.95:
        wnoise = loguniform()
        data = data + wnoise*white(len(data))
    if np.random.uniform(0, 1)>0.95:
        stretch_val = np.random.uniform(0.9, 1.1)
        data = stretch(data, stretch_val)
    if np.random.uniform(0, 1)>0.95:
        pitch_shift_val = np.random.uniform(-6, 6)
        data = pitch_shift(data, n_steps=pitch_shift_val)
    return data

def load_audio_file(file_path, input_length=input_length):
    data = librosa.core.load(file_path, sr=16000)[0] #, sr=16000
    if len(data)>input_length:
        
        
        max_offset = len(data)-input_length
        
        offset = np.random.randint(max_offset)
        
        data = data[offset:(input_length+offset)]
        
        
    else:
        if input_length > len(data):
            max_offset = input_length - len(data)

            offset = np.random.randint(max_offset)
        else:
            offset = 0
        
        
        data = np.pad(data, (offset, input_length - len(data) - offset), "constant")
        
    data = augment(data)
    data = mel_spectrum_db(data)

    return data

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

def generator(file_paths, target_labels, batch_size=16):
    while True:
        file_paths, target_labels = shuffle(file_paths, target_labels)
        
        for batch_files, batch_labels in zip(chunker(file_paths, size=batch_size),
                                             chunker(target_labels, size= batch_size)):

            batch_data = [load_audio_file(fpath) for fpath in batch_files]
            batch_data = np.array(batch_data)[:,:,:,np.newaxis]

            
            yield batch_data, batch_labels