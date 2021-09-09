import os,sys
from decimal import Decimal
import numpy as np
import glob
import importlib
import gc

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle
import gc # Garbage Collector
from numpy import savez_compressed
from joblib import dump, load

from keras.models import load_model
from keras.callbacks import ModelCheckpoint

import keras
import tensorflow as tf

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
        self.last_samples = self.n_input - 1       
        
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
        X= self.__data_generation(list_IDs_temp)
        #print('Yieded batch %d' % index)
        return X

    def on_epoch_end(self):
        #'Updates indexes after each epoch'
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels * self.n_input))
        #y = np.empty((self.batch_size, *self.dim, self.n_channels * self.n_output))
        
        for i, ID in enumerate(list_IDs_temp):  
            inFrom = ID
            inTo = inFrom + self.n_input
            outFrom = inTo
            #outTo = outFrom + self.n_output
            #print(inFrom, inTo, outFrom, outTo)
            X[i] = np.concatenate((self.ds[inFrom : inTo]), axis = 3)
            #y[i] = np.concatenate((self.ds[outFrom : outTo]), axis = 3)
        return X #, y

def isInt(x):
    if x%1 == 0:
        return True
    else:
        return False
     
if __name__ == "__main__":
    SCALER = True
    error = False
    n_args = len(sys.argv)
    dimX = 117
    dimY = 86
    dimZ = 38
    n_output = 10
    n_input = 3
    inputfile = sys.argv[1]
    model_path = sys.argv[2]
    ts_start = sys.argv[3]
    ts_end = sys.argv[4]  
    dims = (dimX, dimY, dimZ)
    n_cells = dimX * dimY * dimZ

    print("From file \"%s\" prediction timesteps from %s to %s with dimensions: " % (inputfile, ts_start, ts_end), dims)
    
    n_channels = 3
    leakyrelu = lambda x: tf.keras.activations.relu(x, alpha=0.1)
    
    ds = []
    filelist = [inputfile]
    for filename in filelist:
            #print(filename)
            dsaux = np.load(filename)
            dsaux = dsaux.f.data
            dsaux = dsaux[-n_input:]
            print(filename, dsaux.shape)
            
            ds.append(dsaux.reshape(dsaux.shape[0] * dsaux.shape[1] * dsaux.shape[2] * dsaux.shape[3], dsaux.shape[4]))
            #print(ds[-1].shape)

    ds = np.array(ds)
    ds = np.concatenate(ds, axis=0)

    # ## Load model
    path = model_path

    pred_ds = []
    axis = ['X', 'Y', 'Z']
    for c in range(n_channels):
        model = load_model(path + "/model%d.hdf5" % c, custom_objects={'<lambda>': leakyrelu})
        model.load_weights(path + "/weights%d.hdf5" % c)
        dsS = np.array(ds[:, c])
        dsS = dsS.reshape((-1, n_cells))
        scaler = load(path + '/scaler%s.joblib' % axis[c])
        dsS = scaler.transform(dsS)
        ds_channel = dsS.reshape((-1,) + dims + (1,))  
        #print("ds_channel", ds_channel.shape)
    
        test = DataGenerator(np.expand_dims(ds_channel, axis=0), batch_size=1, dim=dims, n_channels=1, shuffle=False, observation_samples=n_input, multistep = n_output)  
        pred_output = model.predict(test, verbose=1)
        #print("pred_output ", pred_output.shape)
        
        steps = []
        for step in range(pred_output.shape[-1]):
            steps.append(np.concatenate(pred_output[:,:,:,:,step], axis = 0))
        steps = np.array(steps)
        #print("steps", steps.shape)
        
        dsS = steps.reshape(-1, n_cells)
        dsS = scaler.inverse_transform(dsS)
        pred_ds.append(dsS.reshape((n_output,) + dims + (1,)))
     
    pred_ds = np.squeeze(np.stack(pred_ds, axis=4))#np.array(pred_ds)
    #print("pred_ds", pred_ds.shape)
     
    aux_ts = int(round(float(ts_start) * 100))     
    for i in range(n_output): 
        #print(aux_ts, i)
        #ts = "{:.2f}".format((aux_ts + i) / 100.0)
        aux = round(float((aux_ts + i) / 100.0), 2)       
        if isInt(aux):
            ts = "%d" % int(aux)
        else:
            decimalscnt = str(aux)[::-1].find('.')
            if decimalscnt == 1:
                ts = "{:.1f}".format(aux)
            else:
                ts = "{:.2f}".format(aux)      
        tsU = pred_ds[i].reshape(-1, n_channels)
        path = "ts%sU.npz" % (ts)
        print("Saving file: %s, with shape:" % (path), tsU.shape)
        savez_compressed(path, data=tsU, header=None)            