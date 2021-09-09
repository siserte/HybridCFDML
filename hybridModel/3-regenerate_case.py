#!/usr/bin/env python
# coding: utf-8

import numpy as np
import subprocess as sp
import os, sys
from numpy import savez_compressed
import pathlib  
import shutil  
import glob
from platform import python_version
  
dimCells = (117,86,38)
totalCells = np.prod(dimCells) 

def getEnabledCellsId():
    with open('building', 'r') as file:
        for i, line in enumerate(file):   
            if i == 18:
                meshCells = int(line.strip())
                break

    cellBoolArray = np.zeros((totalCells), dtype=bool)
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
                cellBoolArray[value] = True
                if (cnt == (meshCells + headerSize)):
                    break
    
    #buildingCells = cellBoolArray
    #buildingCells = cellBoolArray.reshape(dimCells,  order='F')
    #buildingCells = buildingCells.reshape(-1,  order='F')
    
    return cellBoolArray
    
def getEnabledMatrix():
    with open('building', 'r') as file:
        for i, line in enumerate(file):   
            if i == 18:
                meshCells = int(line.strip())
                break

    cellBoolArray = np.zeros((totalCells), dtype=bool)
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
                cellBoolArray[value] = True
                if (cnt == (meshCells + headerSize)):
                    break

    matrix = cellBoolArray.reshape(dimCells,  order='C')
    
    return matrix
  
if __name__ == "__main__":
    n_args = len(sys.argv)
    caseVar = 'U'
    path = sys.argv[1]
    #caseId = "{:.2f}".format(float(sys.argv[2]) / 100)
    caseId = sys.argv[2]
    last_of = sys.argv[3]
   
    predictedTs = "ts%s%s.npz" % (caseId, caseVar)

    #print("ERROR - USAGE: python %s <path> <ts>" % (sys.argv[0]))
    #sys.exit(-1)

    print("Regenerating timestep %s in %s (%s)" % (caseId, path, predictedTs))
    
    ds = np.load(predictedTs)
    ds = np.array(ds.f.data, order='C')
    ds = ds.reshape(dimCells + (3,),  order='C')
    ds = ds.reshape((-1, 3),  order='F')
    
    print("Predicted DS shape", ds.shape)
 
    enabledCells = getEnabledCellsId()
    print("Building DS shape", enabledCells.shape)
    #enabledMatrix = getEnabledMatrix()
    #print("Building DS shape", enabledMatrix.shape)
    
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
    print(cnt)  
    
    filename = "%s/%s/U" % (path, str(last_of))
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
         
    dir1 = "%s/%s" % (path, str(last_of))
    dir2 = "%s/%s" % (path, caseId)
    print("Copying %s in %s ..." % (dir1, dir2))
    shutil.copytree(dir1, dir2, dirs_exist_ok = True)      
      
      
    filename = "%s/%s/U" % (path, caseId)      
    print("Writing file %s ..." % filename)
    with open(filename, 'w') as writer:     
        writer.write(header)
        writer.write(values)
        writer.write(footer)
        
    command = "touch %s/%s/prediction" % (path, caseId)
    #print(command)
    child = sp.Popen(command.split(), stdout=sp.PIPE)
    streamdata = child.communicate()[0]
    rc = child.returncode
    
    #COMMENT ONLY FOR TESTING PURPOSES
    os.remove(predictedTs) 