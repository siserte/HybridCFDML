#!/usr/bin/env python
# coding: utf-8
import os
import sys
import shutil  
import subprocess as sp
from platform import python_version
import time
from decimal import Decimal
import numpy as np
import glob
import importlib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle
import gc
from numpy import savez_compressed
from joblib import dump, load
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
import keras
import tensorflow as tf
from multiprocessing import Pool, Manager, cpu_count
m = Manager()

PARALLEL = 0       #0 not runs in parallel. Otherwise, the number of processors to use.
#PARALLEL = cpu_count()

if not PARALLEL:
    from tensorflow.compat.v1.keras.backend import set_session
    config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1,
                            inter_op_parallelism_threads=1,
                            allow_soft_placement=True,
                            device_count = {'CPU': 1})
    session = tf.compat.v1.Session(config=config)
    set_session(session)

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

def is_number(string):
    try:
        float(string)
        return True
    except ValueError:
        return False

def get_first_ts(casename):
    (dirpath, dirnames, filenames) = next(os.walk(casename))
    ids=[]
    for dir in dirnames:
        if (is_number(dir)):
            if (float(dir) >= 0):
                ids.append(dir)
    ids = sorted(ids)
    return ids[-1]
        
def parse_OF_dirname(flt):  
    if isInt(flt):
        string = "%d" % int(flt)
    else:
        decimalscnt = str(flt)[::-1].find('.')
        if decimalscnt == 1:
            string = "{:.1f}".format(flt)
        else:
            string = "{:.2f}".format(flt)
    return string

def writeParamsFile(path, start, end, when):
    filename = path + "/params"
    print("Writing file %s ..." % filename)
    with open(filename, 'w') as f:
        f.write("ts_start\t%d;\n" % start)
        f.write("ts_end\t%d;\n" % end)
        f.write("start\t%s;\n" % when)

def createDirTree(srcpath, dstpath):
    removetime_start = time.time()        
    if os.path.exists(dstpath) and os.path.isdir(dstpath):
        print("Removing directory %s ..." % (dstpath))
        #shutil.rmtree(dstpath)
        command = "rm -rf %s" % (dstpath)
        sp.run(command.split(), capture_output=True) 
    removetime_end = time.time()  
    print("@@@TimeRemoveDir: ", removetime_end - removetime_start)
        
    print("Copying %s in %s ..." % (srcpath, dstpath))
    shutil.copytree(srcpath, dstpath)
    
    command = "touch %s/%s.foam" % (dstpath, dstpath)
    #print("Running %s ..." % (command))
    sp.run(command.split(), capture_output=True) 

def regenerate(caseId, caseDs):
    caseVar = 'U'
    path = dstpath
   
    last_of_str = parse_OF_dirname(round(float(last_of), 2)) 
    
    ds = caseDs.reshape(dimCells + (3,),  order='C')
    ds = ds.reshape((-1, 3),  order='F')
    
    #print("Predicted DS shape", ds.shape)
 
    enabledCells = cellBoolArray
    #print("Building DS shape", enabledCells.shape)
    
    cnt=0
    values = "(\n"
    for idx, cell in enumerate(enabledCells):
        if not cell:
            for c in range(3):
                ds[idx][c] = -1000
            cnt += 1
        else:
            #print(ds[idx])
            cell = "(%f %f %f)\n" % (ds[idx][0], ds[idx][1], ds[idx][2])
            values = values + cell
         
    dir1 = "%s/%s" % (path, last_of_str)
    dir2 = "%s/%s" % (path, caseId)
    #print("Copying %s in %s ..." % (dir1, dir2))
    shutil.copytree(dir1, dir2, dirs_exist_ok = True)      
      
      
    filename = "%s/%s/U" % (path, caseId)      
    #print("Writing file %s ..." % filename)
    with open(filename, 'w') as writer:     
        writer.write(header)
        writer.write(values)
        writer.write(footer)
        
    command = "touch %s/%s/prediction" % (path, caseId)
    child = sp.Popen(command.split(), stdout=sp.PIPE)
    streamdata = child.communicate()[0]
    rc = child.returncode

def callRegenerate(ts):
    caseId = round(float(ts[0]) / 100, 2)
    if isInt(caseId):
        caseId = "%d" % int(caseId)
    else:
        decimalscnt = str(caseId)[::-1].find('.')
        if decimalscnt == 1:
            caseId = "{:.1f}".format(caseId)
        else:
            caseId = "{:.2f}".format(caseId) 
    
    regenerate(caseId, ts[1])
   
def CNN_prediction(n_channels, ds, models, scalers, ts_start, n_output):
    pred_ds = []    
    for c in range(n_channels):
        dsS = np.array(ds[:, c])
        dsS = dsS.reshape((-1, n_cells))
        dsS = scalers[c].transform(dsS)
        ds_channel = dsS.reshape((-1,) + dimCells + (1,))  

        test = DataGenerator(np.expand_dims(ds_channel, axis=0), batch_size=1, dim=dimCells, n_channels=1, shuffle=False, observation_samples=n_input, multistep = n_output)  
        pred_output = models[c].predict(test, verbose=0)
        
        steps = []
        for step in range(pred_output.shape[-1]):
            steps.append(np.concatenate(pred_output[:,:,:,:,step], axis = 0))
        steps = np.array(steps)
        
        dsS = steps.reshape(-1, n_cells)
        dsS = scalers[c].inverse_transform(dsS)
        pred_ds.append(dsS.reshape((n_output,) + dimCells + (1,)))
     
    pred_ds = np.squeeze(np.stack(pred_ds, axis=4))   
    return pred_ds

def mapValuesToCells(x):
    if (float(x)).is_integer():
        ts = str(int(x))
    else:
        ts = str(x)
    #print("Timestep: ", ts)
    scene = getTimestepData(ts) 
    #print("scene", scene[-1])
    meshCells = scene.shape[0]
    cntEmpty = 0
    snapshot = np.array([0,0,0] * n_cells, copy=True, dtype = np.float32).reshape((-1, 3), order='C')
    offset = 0   
    for i in range(n_cells):
        if (cellBoolArray[i]):
            snapshot[i] = scene[i-offset]
        else:
            offset+=1

    ds = np.array(snapshot.reshape(dimCells + (3,)), order='F')
    ds = ds.reshape((-1, 3),  order='C')
    ds = ds.reshape(dimCells + (3,),  order='F')
    shared[dictIds[x]] = ds

def getTimestepData(ts):
    filename = "%s/%s/U" % (casename, ts)
    #print(filename)
    scene = []
    with open(filename) as reader:
        headerlines = 21
        head = [next(reader) for x in range(headerlines)]
        try:
            ncells = int(reader.readline().strip())
        except:
            print("ERROR: ", reader.readline().strip(), "cannot be converted to int. Timestep: %d", ts)
            sys.exit(-1)
        reader.readline()
        for lineid in range(ncells):
            line = reader.readline()
            #print("file", line[:-1])
            scene.append(list(line[1:-2].split(" ")))
            #print("scene", scene[-1])
            #print("\n")
            #sys.exit()
    return np.array(scene, dtype = np.float32)
       
def generate_dataset():
    error = False
    n_args = len(sys.argv)
    n_input = 3
    dimx = 117
    dimy = 86
    dimz = 38
    dimCells = (dimx, dimy, dimz)
    input_path = sys.argv[1]
    output_path = sys.argv[2]

    print("Reading last %d timesteps from \"%s\" writing into \"%s\" with dimensions: " % (n_input, input_path, output_path), dimCells)
    
    path = input_path
    totalCells = np.prod(dimCells) 
    cellBoolArray = np.zeros((totalCells), dtype=bool)
    dictIds = {}

    with open('building', 'r') as file:
        for i, line in enumerate(file):   
            if i == 18:
                meshCells = int(line.strip())
                break 
                
    # Create the boolean array that indicates the building positions
    with open('building', 'r') as file:
        headerSize = 20
        line = file.readline()
        cnt = 1
        while line != '':  # The EOF char is an empty string
            #print(line, end='')
            line = file.readline()
            cnt += 1
            if (cnt > headerSize):
                value = int(line.strip())
                cellBoolArray[value] = True #not a building cell
                if (cnt == (meshCells + headerSize)):
                    break
    #cellBoolArray = cellBoolArray.reshape(dimCells,  order='F')
    #cellBoolArray = cellBoolArray.reshape(-1,  order='C')    
    
    casename = path
    #for casename in glob.glob("case*"):
    
    # Get timesteps
    (dirpath, dirnames, filenames) = next(os.walk(casename))
    ids=[]
    for dir in dirnames:
        if (is_number(dir)):
            if (float(dir) >= 0):
                ids.append(dir)
    ids = sorted(ids)
    ids = ids[-n_input:]
    print("Processing timesteps: ", ids)
    
    # With this dictionary we prevent errors when timesteps do not start in 0 or they are not equally spaced in time.
    for idx, tsId in enumerate(ids):
        dictIds[tsId] = idx    
    #print(dictIds)

    # Map data from VTK to the snapshot
    for i in range(len(ids)):
        shared.append(-1)

    p = Pool(cpu_count())
    #p = Pool(1)
    p.map(mapValuesToCells, ids)
    p.close()
    p.join() 
    
    #scene = np.stack(shared[1:], axis=0)
    scene = np.stack(shared, axis=0)

def loadBuildings():
    totalCells = np.prod(dimCells) 
    cellBoolArray = np.zeros((totalCells), dtype=bool)
    
    with open('building', 'r') as file:
        for i, line in enumerate(file):   
            if i == 18:
                meshCells = int(line.strip())
                break 
                
    # Create the boolean array that indicates the building positions
    with open('building', 'r') as file:
        headerSize = 20
        line = file.readline()
        cnt = 1
        while line != '':  # The EOF char is an empty string
            #print(line, end='')
            line = file.readline()
            cnt += 1
            if (cnt > headerSize):
                value = int(line.strip())
                cellBoolArray[value] = True #not a building cell
                if (cnt == (meshCells + headerSize)):
                    break
    return cellBoolArray

def loadCNN():
    leakyrelu = lambda x: tf.keras.activations.relu(x, alpha=0.1)
    models = []
    scalers = []
    for c in range(n_channels):
        models.append(load_model(model_path + "/model%d.hdf5" % c, custom_objects={'<lambda>': leakyrelu}))
        models[-1].load_weights(model_path + "/weights%d.hdf5" % c)
        scalers.append(load(model_path + '/scaler%s.joblib' % axis[c]))
    return models, scalers

if __name__ == "__main__":
    starttotal = time.time()
    print('Python', python_version())  
    
    dimX = 117
    dimY = 86
    dimZ = 38
    dstpath = "output_dataset_200"
    npzfile = "tmp_dataset"
    n_input = 3
    dimCells = (dimX, dimY, dimZ)
    n_cells = dimX * dimY * dimZ
    axis = ['X', 'Y', 'Z']
    n_channels = 3
    
    ### Configure these:
    n_output = 10
    of_tss = 10
    of_tss_ini = 3
    model_path = "MODCHAN10-0-200"
    grain = 0.01
    DRYRUN = False       #Copy ts from an OF simulation instead of simulating them.
    OFFIRST = True    #Start the execution simulating instead of predicting.
    CONTINUE = False     #Continue a previous execution instead of start from scratch.
    BREAKFIRST = False   #Run a single iteration: simulate (if OFFIRST), generate, predict, and regenerate.
    #srcpath = 'CASE_test_3.5_0.5_lite'
    #srcpath = 'CASE_test_3.5_0.5'
    srcpath = 'CASE_3.5_0.5_par'
    last_ts = 7.01
    ofcommand = "pimpleFoam"

    cellBoolArray = loadBuildings()

    models, scalers = loadCNN()

    print("----------------------------------------------------------------------")
    if CONTINUE:
       srcpath = dstpath
       OFFIRST = False
    else:
        createDirTree(srcpath, dstpath)
        
    first_ts = get_first_ts(srcpath)
    
    filename = "%s/%s/U" % (dstpath, str(first_ts))
    header = ""
    footer = ""
    headerlines = 21 
    with open(filename, 'r') as reader:
        line = reader.readline()
        cntline = 0
        cntarray = 0
        while line:
            if cntline <= headerlines:
                header = header + line
                if cntline == headerlines:
                    ncells = int(line.strip())
                    reader.readline()
                    cntline += 1
            elif cntline > (headerlines + ncells +1):
                footer = footer + line
            line = reader.readline()
            cntline += 1
    
    ts_curr = "{:.2f}".format(round(float(first_ts) + grain, 2))
    last_of = "{:.2f}".format(round(float(first_ts), 2))
    
    print("----------------------------------------------------------------------")
    end = time.time()
    print("@@@TimeInit: ", end - starttotal, "ts: ", ts_curr)
    start = time.time()       
    print("----------------------------------------------------------------------")

    while True:
        if OFFIRST:
            ts_end = "{:.2f}".format(float(ts_curr) + ((of_tss_ini-2) * grain))
            with open(dstpath + "/params", "w") as paramsfile:
                paramsfile.write("ts_start\t%s;\nts_end\t%s;\n" % (ts_curr, ts_end))
            if PARALLEL:
                command = "decomposePar -case %s -force" % (dstpath)
                print("%s" % (command))
                child = sp.Popen(command.split(), stdout=sp.PIPE)
                streamdata = child.communicate()[0]
                command = "mpirun -n %d %s -case %s -parallel" % (PARALLEL, ofcommand, dstpath)
            else:
                command = "%s -case %s" % (ofcommand, dstpath)
            print("\n$ %s (from timestep %s to %s)" % (command, ts_curr, ts_end))
            child = sp.Popen(command.split(), stdout=sp.PIPE)
            streamdata = child.communicate()[0]
            rc = child.returncode
            if(rc != 0):
                sys.exit()
            #print(streamdata.decode())
            
            if PARALLEL:
                command = "./parReconstructPar.sh -c %s -n %d" % (dstpath, PARALLEL)
                print("%s" % (command))
                child = sp.Popen(command.split(), stdout=sp.PIPE)
                streamdata = child.communicate()[0]
            
            ts_curr = "{:.2f}".format(round(float(ts_end) + grain, 2))
            last_of = ts_end
            OFFIRST = False
            
            print("----------------------------------------------------------------------")     
            end = time.time()
            print("@@@TimeFirstSimulation: ", end - start, "# start from ts: ", ts_curr)
            start = time.time()
            print("----------------------------------------------------------------------")
        
        else:
            if float(ts_curr) < last_ts:
                ts_end = "{:.2f}".format(float(ts_curr) + ((of_tss-1) * grain))
            else:
                ts_end = "{:.2f}".format(float(ts_curr))
                BREAKFIRST = True
                break
            
            with open(dstpath + "/params", "w") as paramsfile:
                paramsfile.write("ts_start\t%s;\nts_end\t%s;\n" % (ts_curr, ts_end))
                
            if PARALLEL:
                command = "decomposePar -case %s -force" % (dstpath)
                print("%s" % (command))
                child = sp.Popen(command.split(), stdout=sp.PIPE)
                streamdata = child.communicate()[0]
                command = "mpirun -n %d %s -case %s -parallel" % (PARALLEL, ofcommand, dstpath)
            else:
                command = "%s -case %s" % (ofcommand, dstpath)
            
            print("\n$ %s (from timestep %s to %s)" % (command, ts_curr, ts_end))
            child = sp.Popen(command.split(), stdout=sp.PIPE)
            streamdata = child.communicate()[0]
            rc = child.returncode
            if(rc != 0):
                sys.exit()    
                
            if PARALLEL:
                command = "./parReconstructPar.sh -c %s -n %d" % (dstpath, PARALLEL)
                print("%s" % (command))
                child = sp.Popen(command.split(), stdout=sp.PIPE)
                streamdata = child.communicate()[0]
                    
            ts_curr = "{:.2f}".format(round(float(ts_end) + grain, 2))
            last_of = ts_end
            
            print("----------------------------------------------------------------------")
            end = time.time()
            print("@@@TimeSimulation: ", end - start)
            start = time.time()
            print("----------------------------------------------------------------------") 
            
        #last_of_str = parse_OF_dirname(round(float(last_of), 2))
        casename = dstpath
        (dirpath, dirnames, filenames) = next(os.walk(casename))
        ids=[]
        for dir in dirnames:
            if (is_number(dir)):
                if (float(dir) >= 0):
                    ids.append(dir)
        ids = sorted(ids)
        ids = ids[-n_input:]

        # With this dictionary we prevent errors when timesteps do not start in 0 or they are not equally spaced in time.
        dictIds = {}
        for idx, tsId in enumerate(ids):
            dictIds[tsId] = idx    

        # Map data from VTK to the snapshot
        shared = m.list()
        for i in range(len(ids)):
            shared.append(-1)

        if PARALLEL:
            p = Pool(PARALLEL)
        else:
            p = Pool(1)
        p.map(mapValuesToCells, ids)
        p.close()
        p.join() 
        
        scene = np.stack(shared, axis=0)
        
        print("----------------------------------------------------------------------")     
        end = time.time()
        print("@@@TimeDataset: ", end - start, "# processed ts: ", ids)
        start = time.time()
        print("----------------------------------------------------------------------")

        dsaux = scene[-n_input:] 
        ds = dsaux.reshape(dsaux.shape[0] * dsaux.shape[1] * dsaux.shape[2] * dsaux.shape[3], dsaux.shape[4])
        dspred = CNN_prediction(n_channels, ds, models, scalers, ts_curr, n_output)

        print("----------------------------------------------------------------------")
        end = time.time()
        print("@@@TimePrediction: ", end - start)
        start = time.time()
        print("----------------------------------------------------------------------")

        tslist = []
        auxlist = []
        aux_ts = int(round(float(ts_curr) * 100))
        for i in range(dspred.shape[0]):
            tslist.append((aux_ts + i, dspred[i]))
            auxlist.append((aux_ts + i) / 100)
              
        if PARALLEL:
            #p = Pool(min(n_output, cpu_count()))
            p = Pool(min(n_output, PARALLEL))
        else:
            p = Pool(1)
        p.map(callRegenerate, tslist)
        p.close()
        p.join()      
        #for ts in tslist:
        #    callRegenerate(ts)
        
        ts_end = "{:.2f}".format(round(float(ts_curr) + (n_output - 1) * grain, 2))
        ts_curr = ts_end
        
        print("----------------------------------------------------------------------")
        end = time.time()
        print("@@@TimeRegenerate: ",  end - start, "# writen ts: ", auxlist)
        start = time.time()
        print("----------------------------------------------------------------------")
        
        """
        if float(ts_curr) < last_ts:
            ts_end = "{:.2f}".format(float(ts_curr) + ((of_tss-1) * grain))
        else:
            ts_end = "{:.2f}".format(float(ts_curr))
            BREAKFIRST = True
        
        with open(dstpath + "/params", "w") as paramsfile:
            paramsfile.write("ts_start\t%s;\nts_end\t%s;\n" % (ts_curr, ts_end))
            
        if PARALLEL:
            command = "decomposePar -case %s -force" % (dstpath)
            print("%s" % (command))
            child = sp.Popen(command.split(), stdout=sp.PIPE)
            streamdata = child.communicate()[0]
            command = "mpirun -n %d pimpleFoam -case %s -parallel" % (PARALLEL, dstpath)
        else:
            command = "pimpleFoam -case %s" % dstpath
        
        print("\n$ %s (from timestep %s to %s)" % (command, ts_curr, ts_end))
        child = sp.Popen(command.split(), stdout=sp.PIPE)
        streamdata = child.communicate()[0]
        rc = child.returncode
        if(rc != 0):
            sys.exit()    
            
        if PARALLEL:
            command = "./parReconstructPar.sh -c %s -n %d" % (dstpath, PARALLEL)
            print("%s" % (command))
            child = sp.Popen(command.split(), stdout=sp.PIPE)
            streamdata = child.communicate()[0]
                
        print("----------------------------------------------------------------------")
        end = time.time()
        if float(ts_curr) < last_ts:
            print("@@@TimeSimulation: ", end - start)
        else:
            print("@@@TimeLastSimulation: ", end - start)
        start = time.time()
        print("----------------------------------------------------------------------")
        
        print("----------------------------------------------------------------------")
        end = time.time()
        ts_curr = "{:.2f}".format(round(float(ts_end), 2))
        last_of = ts_end
        print("@@@Time accumulated until ts %s: " % (ts_curr), end - starttotal)
        print("----------------------------------------------------------------------")
        """
        
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")   
        ts_curr = "{:.2f}".format(round(float(ts_end) + grain, 2))
        print("\nNEXT TIMESTEP: %s\n" % ts_curr)
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        
        if BREAKFIRST:
            #if float(ts_curr) > 1.26:
            break
        
    print("----------------------------------------------------------------------")
    endtotal = time.time()
    print("@@@TimeTotal: ", endtotal - starttotal)
    with open(dstpath + "/log", "w") as file:
        file.write(streamdata.decode())
    print("----------------------------------------------------------------------")
