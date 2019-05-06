# importing the utility functions
from utils_1d import input_to_target, load_audio_file, generator
# 
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split

# keras components for constructing the model
import tensorflow as tf
from tensorflow.keras import optimizers, losses, activations, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from tensorflow.keras.layers import (Dense, Input, Dropout, Convolution1D, 
MaxPool1D, GlobalMaxPool1D, GlobalAveragePooling1D, concatenate)
from tensorflow.keras.applications.xception import Xception
import gc



def build_1d_model(input_length, nclass):
    
    input_wave = Input(shape=(input_length, 1))
    
    # convolution block 1
    conv_1 = Convolution1D(16, kernel_size=9, activation=activations.relu, padding="valid")(input_wave)
    conv_1 = Convolution1D(16, kernel_size=9, activation=activations.relu, padding="valid")(conv_1)
    conv_1 = MaxPool1D(pool_size=16)(conv_1)
    conv_1 = Dropout(rate=0.1)(conv_1)
    
    # convolution block 2
    conv_2 = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid")(conv_1)
    conv_2 = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid")(conv_2)
    conv_2 = MaxPool1D(pool_size=8)(conv_2)
    conv_2 = Dropout(rate=0.1)(conv_2)

    
    # convolution block 4
    conv_3 = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid")(conv_2)
    conv_3 = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid")(conv_3)
    conv_3 = MaxPool1D(pool_size=4)(conv_3)
    conv_3 = Dropout(rate=0.1)(conv_3)

    
    # convolution block 5
    conv_4 = Convolution1D(256, kernel_size=3, activation=activations.relu, padding="valid")(conv_3)
    conv_4 = Convolution1D(256, kernel_size=3, activation=activations.relu, padding="valid")(conv_4)
    conv_4 = GlobalMaxPool1D()(conv_4)
    conv_4 = Dropout(rate=0.2)(conv_4)

    # dense block 1
    dense_1 = Dense(64, activation=activations.relu)(conv_4)
    dense_1 = Dense(1028, activation=activations.relu)(dense_1)
    dense_1 = Dense(nclass, activation=activations.softmax)(dense_1)

    model = models.Model(inputs=input_wave, outputs=dense_1)
    opt = optimizers.Adam(0.00001)

    model.compile(optimizer=opt, loss=losses.sparse_categorical_crossentropy, metrics=['acc'])
    model.summary()
    return model

# train model
def train_audio(train_files, train_labels,
            val_files, val_labels, input_length = 64000, nclass = 10, epochs = 20, batch_size=32):

    model = build_1d_model(input_length=input_length, nclass=n_class)
    model.fit_generator(generator(train_files, train_labels), steps_per_epoch=len(train_files)//batch_size, epochs=epochs,

                        validation_data=generator(val_files, val_labels), 
                        validation_steps=len(val_files)//batch_size,
                        use_multiprocessing=True, max_queue_size=1,
                        callbacks=[ModelCheckpoint("models/model_1d.h5",
                                                    monitor="val_acc", save_best_only=True),
                                    EarlyStopping(patience=5, monitor="val_acc")])


    model.save("models/model_1d.h5")

sampling_freq = 16000
duration = 4
input_length = sampling_freq*duration
batch_size = 32
train_file_to_label = input_to_target()
input_files = train_file_to_label['train_file_paths'].values
target_labels = train_file_to_label['class_int_encode'].values
train_files, val_files, train_labels, val_labels = train_test_split(input_files, target_labels,
                                                                   test_size=0.15, random_state=42)
n_class= len(train_file_to_label['Class'].unique())

train_audio(train_files, train_labels,
                val_files, val_labels, input_length = input_length, nclass = 10, epochs = 1)