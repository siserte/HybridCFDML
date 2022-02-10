#!/usr/bin/env python
# coding: utf-8

import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
#os.environ["CUDA_VISIBLE_DEVICES"] = "3"
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from decimal import Decimal
import numpy as np
import glob
import importlib
import gc
import matplotlib.pyplot as plt
#from livelossplot import PlotLossesKeras
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle
import gc # Garbage Collector
import pandas as pd
from numpy import savez_compressed
from joblib import dump, load
import h5py

from keras.models import load_model
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential, load_model
from keras.layers import LSTM, TimeDistributed
from keras.layers.core import Dense, Flatten, Dropout, RepeatVector, Reshape
from keras.layers.convolutional import Conv3D, Conv3DTranspose, MaxPooling3D 
from keras.callbacks import ModelCheckpoint

from platform import python_version
import keras
import tensorflow as tf

from tensorflow.compat.v1.keras.backend import set_session
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # True dynamically grow the memory used on the GPU
sess = tf.compat.v1.Session(config=config)
set_session(sess)

from tensorflow.python.client import device_lib
def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

print('Notebook running on Python', python_version())
print('Numpy version', np.version.version)
print('Scikit-learn version {}.'.format(sklearn.__version__))
print('Keras version ', keras.__version__,'and TensorFlow', tf.__version__, '(CUDA:', tf.test.is_built_with_cuda(), '- GPUs available:', get_available_gpus(), ')')

dimX, dimY, dimZ = 117, 86, 38
n_cells = dimX * dimY * dimZ
dims = (dimX, dimY, dimZ)
n_channels = 1
n_input = 3
n_output = 10
n_batch = 8
leakyrelu = lambda x: tf.keras.activations.relu(x, alpha=0.1)

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_datasets, batch_size=128, dim=(32,32,32), n_channels=3, shuffle=True, observation_samples=3, multistep=1):
        #print('Generator Initialization')
        self.dim = dim
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.n_input = observation_samples      
        self.n_output = multistep
        self.last_samples = self.n_input - 1 + self.n_output      
        
        self.indexes = []
        cnt = 0
        self.total_samples = 0
        for case in list_datasets:
            samples = len(case) - self.last_samples
            self.total_samples += samples
            for sample in range(samples):
                self.indexes.append(cnt)
                cnt += 1
            cnt += self.last_samples
        
        self.ds = np.concatenate((list_datasets), axis = 0)
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.total_samples / self.batch_size))
  
    def __getitem__(self, batch_index):          
        'Generate one batch of data through dataset'
        list_IDs_temp = self.indexes[batch_index * self.batch_size : (batch_index+1) * self.batch_size]
        #print(index, list_IDs_temp)
        # Generate data
        X, y = self.__data_generation(list_IDs_temp)
        #print('Yieded batch %d' % index)
        return X, y

    def on_epoch_end(self):
        #'Updates indexes after each epoch'
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels * self.n_input))
        y = np.empty((self.batch_size, *self.dim, self.n_channels * self.n_output))
        
        for i, ID in enumerate(list_IDs_temp):  
            inFrom = ID
            inTo = inFrom + self.n_input
            outFrom = inTo
            outTo = outFrom + self.n_output
            
            X[i] = np.concatenate((self.ds[inFrom : inTo]), axis = 3)
            y[i] = np.concatenate((self.ds[outFrom : outTo]), axis = 3)
        return X, y

tsini=0
tsend=150
basepath = "/home/bsc21/bsc21334/SCRATCH/aina-data/"
pathnpz = basepath + "/NPZs/"
targetname = "/case3.5.npz"
path = basepath + "/MODCHAN%d-%d-%d/" % (n_output, tsini, tsend)
if not os.path.exists(path):
    os.mkdir(path)
datapath = basepath + "/DATA-%d-%d/" % (tsini, tsend)

with h5py.File(datapath + "data.h5","r") as hf:
    ds_ready = hf["dataset"][:]

print('Dataset size in memory: %0.2f GB' % (ds_ready.nbytes / 1024**3))
np.info(ds_ready)

n_epochs=100
n_batch = 8
trainratio = -3
for c in range(2,3):
    model = Sequential()
    model.add(Conv3D(64, kernel_size=(3, 3, 3), activation=leakyrelu, padding="same", data_format="channels_last", input_shape=(dimX, dimY, dimZ, n_input)))
    model.add(Conv3D(128, kernel_size=(3, 3, 3), activation=leakyrelu, padding="same"))
    #model.add(Conv3D(256, kernel_size=(3, 3, 3), activation=leakyrelu, padding="same"))
    #model.add(Conv3D(128, kernel_size=(3, 3, 3), activation=leakyrelu, padding="same"))
    model.add(Conv3D(64, kernel_size=(3, 3, 3), activation=leakyrelu, padding="same"))
    model.add(Dense(n_output))
    model.compile(loss='mae', optimizer='adam')#, metrics=[rmse, 'mae', 'mape'])
    model.save(path + "/model%d.hdf5" %  c)
    
    train = DataGenerator(ds_ready[c][:trainratio], 
    #train = DataGenerator(ds_ready[c], 
                      batch_size=n_batch, dim=dims, n_channels=n_channels, 
                      shuffle=True, observation_samples=n_input, multistep=n_output)
    validation = DataGenerator(ds_ready[c][trainratio:], 
                      batch_size=n_batch, dim=dims, n_channels=n_channels, 
                      shuffle=True, observation_samples=n_input, multistep=n_output)
    #print(len(train))
    #print(len(validation))

    checkpoint_file = path + "/weights%d.hdf5" % c
    earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, min_delta=0.0001, verbose=1, restore_best_weights = True)
    checkpoint = ModelCheckpoint(checkpoint_file, monitor='val_loss', verbose=1, save_best_only=True)
    callbacks_list = [checkpoint, earlystop]

    H = model.fit(train, validation_data=validation, epochs=n_epochs, verbose=1, shuffle=True, callbacks=callbacks_list)
    #H = model.fit(train, epochs=n_epochs, verbose=1, shuffle=True)#, callbacks=callbacks_list)
    print("End of channel", c)
    model.save_weights(checkpoint_file)

    hist = H.history
    plt.style.use("ggplot")
    plt.figure(figsize=(16,8))
    plt.plot(hist["loss"], label="Loss (MSE)")
    plt.plot(hist["val_loss"], label="Val Loss (MSE)")
    plt.title("Training Errors on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Error")
    plt.legend()
    plt.ylim(0,0.1)
    #plt.xlim(0,8)
    plt.savefig("plot%d.png" % c)
    #plt.show()
