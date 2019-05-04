# basic library imports
import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import random
from tqdm import tqdm

# encoding the classes
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# librosa for audion feature extraction
import librosa
import gc
import pickle
import random
from multiprocessing import Pool
from PIL import Image
from random import shuffle
from sklearn.model_selection import train_test_split

# plotly libraries
import plotly.graph_objs as go
from plotly import tools
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
import plotly_express as px


# keras components for constructing the model
import tensorflow as tf
from tensorflow.keras import optimizers, losses, activations, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from tensorflow.keras.layers import (Dense, Input, Dropout, Convolution1D, 
MaxPool1D, GlobalMaxPool1D, GlobalAveragePooling1D, concatenate)

print(librosa.core.load('data/train/Train/1036.wav', sr=16000, duration=5.0))