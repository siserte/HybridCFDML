#!/usr/bin/env python
# coding: utf-8
import os
import sys
import numpy as np
import glob
import importlib
import pathlib
import sklearn
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
from numpy import savez_compressed
from platform import python_version

from multiprocessing import Pool, Manager, cpu_count
m = Manager()
shared = m.list()

def is_number(string):
    try:
        float(string)
        return True
    except ValueError:
        return False

def mapValuesToCells(x):
    if (x).is_integer():
        ts = str(int(x))
    else:
        ts = str(x)
    #print("Timestep: ", ts)
    scene = getTimestepData(ts) 
    #print("scene", scene[-1])
    meshCells = scene.shape[0]
    cntEmpty = 0
    snapshot = np.array([0,0,0] * totalCells, dtype=np.float32).reshape((-1, 3), order='C')
    offset = 0   
    for i in range(totalCells):
        if (cellBoolArray[i]):
            snapshot[i] = scene[i-offset]
        else:
            offset+=1
            
    ds = np.array(snapshot.reshape(dimCells + (3,)), order='C')
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
            print("ERROR: ", reader.readline().strip(), "cannot be converted to int. Timestep: %s in %s" % (ts, filename))
            sys.exit(-1)
        reader.readline()
        for lineid in range(ncells):
            line = reader.readline()
            scene.append(list(line[1:-2].split(" ")))
            
    return np.array(scene, dtype = np.float32)
   
if __name__ == "__main__":
    error = False
    n_args = len(sys.argv)
    dimx = 117
    dimy = 86
    dimz = 38
    ts_start = 50
    ts_end = 501
    
    cpus = 1
    if (n_args == 3 or n_args == 4):
        input_path = sys.argv[1]
        output_path = sys.argv[2]
        if (n_args == 4):
            cpus = int(sys.argv[3])
    else:
        error = True
    
    dimCells = (dimx, dimy, dimz)
    if error:
        #[siserte@srvchiva ainaBuildingsV3]$ python 2-generate_datasets.py experiments/CASE_3/ case3
        print("ERROR - USAGE: python %s <case_dir_path> <npz_file_path> <ts_start> <ts_end> <dimX> <dimY> <dimZ>" % (sys.argv[0]))
        sys.exit(-1)
    else:
        print("Reading case from \"%s\"[%d:%d] writing into \"%s\" with dimensions: " % (input_path, ts_start, ts_end, output_path), dimCells)
    
    path = input_path
    totalCells = np.prod(dimCells) 
    cellBoolArray = np.zeros((totalCells), dtype=bool)
    dictIds = {}

    with open('/home/siserte/ainaBuildingsV4/building', 'r') as file:
        for i, line in enumerate(file):   
            if i == 18:
                meshCells = int(line.strip())
    
    # Create the boolean array that indicates the building positions
    with open('/home/siserte/ainaBuildingsV4/building', 'r') as file:
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
    
    # Get timesteps
    (dirpath, dirnames, filenames) = next(os.walk(casename))
    ids=[]
    for dir in dirnames:
        if (is_number(dir)):
            num = float(dir)
            if (num >= 0):
                ids.append(num)
    ids = sorted(ids)
    print(len(ids))
    ids = ids[ts_start:ts_end+1]
    print("Processing timesteps: ", ids[0], ids[-1])
    
    # With this dictionary we prevent errors when timesteps do not start in 0 or they are not equally spaced in time.
    for idx, tsId in enumerate(ids):
        dictIds[tsId] = idx    
    print(dictIds)

    # Map data from VTK to the snapshot
    for i in range(len(ids)):
        shared.append(-1)

    p = Pool(cpus)
    p.map(mapValuesToCells, ids)
    p.close()
    p.join() 
    
    scene = np.stack(shared, axis=0)

    casename = "/home/siserte/ainaBuildingsV4/NPZs/" + output_path
    print("Saving file: %s.npz, with shape:" % (casename), scene.shape)
    savez_compressed(casename, data=scene, header=None)
    #print("Done!")