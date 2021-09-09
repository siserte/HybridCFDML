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
from sklearn.decomposition import TruncatedSVD
from joblib import dump, load
from numpy import savez_compressed
from platform import python_version

#print('Notebook running on Python', python_version())
#print('Numpy version', np.version.version)
#print('Scikit-learn version {}'.format(sklearn.__version__))

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
    if (float(x)).is_integer():
        ts = str(int(x))
    else:
        ts = str(x)
    #print("Timestep: ", ts)
    scene = getTimestepData(ts) 
    #print("scene", scene[-1])
    meshCells = scene.shape[0]
    cntEmpty = 0
    snapshot = np.array([0,0,0] * totalCells, copy=True, dtype = np.float32).reshape((-1, 3), order='C')
    offset = 0   
    for i in range(totalCells):
        if (cellBoolArray[i]):
            snapshot[i] = scene[i-offset]
        else:
            offset+=1

    #shared[dictIds[x]] = np.array(snapshot.reshape(dimCells + (3,)), order='C', dtype=np.float32)
    
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
   
if __name__ == "__main__":
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

    #casename = "case3_ts-16-20"
    casename = output_path
    print("Saving file: %s.npz, with shape:" % (casename), scene.shape)
    savez_compressed(casename, data=scene, header=None)
    #print("Done!")
    
    
    #UNCOMMENT THIS ONLY FOR TESTING PURPOSES
    """
    for idx, ts in enumerate(scene):
        path = "ts%dU" % (idx + 20)
        tsU = ts.reshape((-1, 3), order='C')
        print("Saving file: %s, with shape:" % (path), tsU.shape)
        savez_compressed(path, data=tsU, header=None) 
    """
