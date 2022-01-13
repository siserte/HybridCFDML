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
from multiprocessing import Pool, cpu_count  
  
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
    
    buildingCells = cellBoolArray
    
    return buildingCells
  
def regenerateTs(caseId):
    ds = dsaux[caseId].reshape(-1, 3)
    #print(ds.shape)
    cellsArray = []
    for idx, cell in enumerate(enabledCells):
        if cell:
            cellsArray.append(ds[idx])
          
    ds = np.array(cellsArray)
    #print("OpenFOAM DS shape", ds.shape)    

    filename = "%s/0.01/U" % (path)
    header = ""
    footer = ""
    values = "(\n"
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
            else:
                cell = "(%f %f %f)\n" % (ds[cntarray][0], ds[cntarray][1], ds[cntarray][2])
                values = values + cell
                cntarray += 1
            line = reader.readline()
            cntline += 1


    dir1 = "%s/0" % (path)
    dir2 = "%s/%d" % (dirpath, caseId)
    #print("Copying %s in %s ..." % (dir1, dir2))
    shutil.copytree(dir1, dir2, dirs_exist_ok = True)      
      
    filename = "%s/%d/U" % (dirpath, caseId)      
    print("Writing file %s ..." % filename)
    with open(filename, 'w') as writer:     
        writer.write(header)
        writer.write(values)
        writer.write(footer)  
 
if __name__ == "__main__":   
    dirpath = "prediction_multistep"
    filename = "NPZs/case3.6.npz" 
    path = "CASE_base/"
    dsaux = np.load(filename)
    dsaux = dsaux.f.data
    print(dsaux.shape)
    
    timesteps = dsaux.shape[0]
    enabledCells = getEnabledCellsId()
    print("Building DS shape", enabledCells.shape)
    
    print(timesteps)
    
    p = Pool(cpu_count())
    #p = Pool(1)
    p.map(regenerateTs, range(0, timesteps))
    p.close()
    p.join() 
     
    command = "cp -r CASE_base/0 CASE_base/constant CASE_base/system %s" % (dirpath)
    print(command)
    sp.run(command.split(), capture_output=True)
    
    command = "touch %s/%s.foam" % (dirpath, dirpath)
    print(command)
    sp.run(command.split(), capture_output=True)
    
    command = "cp -r %s /tmp" % (dirpath)
    print(command)
    sp.run(command.split(), capture_output=True)