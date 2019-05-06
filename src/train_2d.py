# importing the utility functions
from utils_2d import input_to_target, audio_normalization, load_audio_file, generator
# 
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split

# keras components for constructing the model
import tensorflow as tf
from tensorflow.keras import optimizers, losses, activations, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from tensorflow.keras.layers import (Dense, Input, Dropout, Convolution2D, 
MaxPool2D, GlobalMaxPool2D, GlobalAveragePooling2D, concatenate)
from tensorflow.keras.applications.xception import Xception

def build_2d_model(input_shape = (126, 320, 1), nclass = 10):
       
    input_wave = Input(shape=input_shape)

    xception = Xception(input_shape=input_shape, weights=None, include_top=False)

    x = xception(input_wave)
    x = GlobalMaxPool2D()(x)
    x = Dropout(rate=0.1)(x)

    x = Dense(128, activation=activations.relu)(x)
    x = Dense(nclass, activation=activations.softmax)(x)

    model = models.Model(inputs=input_wave, outputs=x)
    opt = optimizers.Adam()

    model.compile(optimizer=opt, loss=losses.sparse_categorical_crossentropy, metrics=['acc'])
    model.summary()
    return model

def train_audio(train_files, train_labels,
                val_files, val_labels, input_shape=(126,320,1), nclass = 10, batch_size= 16, epochs = 20):
    
    model = build_2d_model(input_shape=input_shape, nclass=n_class)
    
    model.fit_generator(generator(train_files, train_labels), steps_per_epoch=len(train_files)//batch_size, epochs=epochs,

                        validation_data=generator(val_files, val_labels), 
                        validation_steps=len(val_files)//batch_size,
                        use_multiprocessing=True, max_queue_size=1,
                        callbacks=[ModelCheckpoint("./models/model_2d.h5",
                                                   monitor="val_acc", save_best_only=True),
                                   EarlyStopping(patience=5, monitor="val_acc")])


    model.save("./models/model_2d.h5")


sampling_freq = 16000
duration = 4
input_length = sampling_freq*duration
batch_size = 16
train_file_to_label = input_to_target()
input_files = train_file_to_label['train_file_paths'].values
target_labels = train_file_to_label['class_int_encode'].values
train_files, val_files, train_labels, val_labels = train_test_split(input_files, target_labels,
                                                                   test_size=0.15, random_state=42)
n_class= len(train_file_to_label['Class'].unique())

train_audio(input_length, train_files, train_labels,
                val_files, val_labels, nclass = 10, epochs = 1)