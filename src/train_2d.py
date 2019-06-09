# importing the utility functions
from utils_2d import input_to_target, audio_normalization, load_audio_file, generator

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# keras components for constructing the model
import tensorflow as tf
from tensorflow.keras import optimizers, losses, activations, models
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateScheduler,
)
from tensorflow.keras.layers import (
    Dense,
    Input,
    Dropout,
    Convolution2D,
    MaxPool2D,
    GlobalMaxPool2D,
    GlobalAveragePooling2D,
    concatenate,
)
from tensorflow.keras.applications.xception import Xception

input_shape = [126, 320, 1]
nclass = 10
epochs = 25
batch_size = 32


def build_2d_model(input_shape=input_shape, nclass=nclass):

    input_wave = Input(shape=input_shape)

    xception = Xception(input_shape=input_shape, weights=None, include_top=False)

    x = xception(input_wave)
    x = GlobalMaxPool2D()(x)
    x = Dropout(rate=0.1)(x)

    x = Dense(128, activation=activations.relu)(x)
    x = Dense(nclass, activation=activations.softmax)(x)

    model = models.Model(inputs=input_wave, outputs=x)
    opt = optimizers.Adam()

    model.compile(
        optimizer=opt, loss=losses.sparse_categorical_crossentropy, metrics=["acc"]
    )
    model.summary()
    return model


def train_audio(
    train_files,
    train_labels,
    val_files,
    val_labels,
    input_shape=input_shape,
    nclass=nclass,
    epochs=epochs,
):

    model = build_2d_model(input_shape=input_shape, nclass=nclass)

    history = model.fit_generator(
        generator(train_files, train_labels),
        steps_per_epoch=len(train_files) // batch_size,
        epochs=epochs,
        validation_data=generator(val_files, val_labels),
        validation_steps=len(val_files) // batch_size,
        use_multiprocessing=True,
        max_queue_size=1,
        callbacks=[
            ModelCheckpoint(
                "./models/model_2d.h5", monitor="val_acc", save_best_only=True
            ),
            EarlyStopping(patience=5, monitor="val_acc"),
        ],
    )

    model.save_weights("./models/model_2d.h5")
    plt.plot(history.history["acc"])
    plt.plot(history.history["val_acc"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper left")
    plt.savefig("./results/'acc_model_2d.png", dpi=300)
    plt.show()
    # summarize history for loss
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.savefig("./results/'loss_model_2d.png", dpi=300)
    plt.legend(["train", "test"], loc="upper left")
    plt.show()


train_file_to_label = input_to_target()
input_files = train_file_to_label["train_file_paths"].values
target_labels = train_file_to_label["class_int_encode"].values
train_files, val_files, train_labels, val_labels = train_test_split(
    input_files, target_labels, test_size=0.15, random_state=42
)
n_class = len(train_file_to_label["Class"].unique())

train_audio(
    train_files,
    train_labels,
    val_files,
    val_labels,
    input_shape=input_shape,
    nclass=nclass,
    epochs=epochs,
)
