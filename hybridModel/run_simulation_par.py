#!/usr/bin/env python
# coding: utf-8
import os
import sys
import shutil  
import subprocess as sp
from platform import python_version
from multiprocessing import Pool, cpu_count
import time

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

def isInt(x):
    if x%1 == 0:
        return True
    else:
        return False
        
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
    if os.path.exists(dstpath) and os.path.isdir(dstpath):
        print("Removing directory %s ..." % (dstpath))
        #shutil.rmtree(dstpath)
        command = "rm -rf %s" % (dstpath)
        sp.run(command.split(), capture_output=True) 

    print("Copying %s in %s ..." % (srcpath, dstpath))
    shutil.copytree(srcpath, dstpath)
    
    command = "touch %s/%s.foam" % (dstpath, dstpath)
    #print("Running %s ..." % (command))
    sp.run(command.split(), capture_output=True) 

def callRegenerate(ts):
    #caseId = "{:.2f}".format(round(float(ts) / 100, 2))
    caseId = round(float(ts) / 100, 2)
    if isInt(caseId):
        caseId = "%d" % int(caseId)
    else:
        decimalscnt = str(caseId)[::-1].find('.')
        if decimalscnt == 1:
            caseId = "{:.1f}".format(caseId)
        else:
            caseId = "{:.2f}".format(caseId) 
    command = "python 3-regenerate_case.py %s %s %s" % (dstpath, caseId, last_of_str)
    print("Running %s ..." % (command))
    child = sp.Popen(command.split(), stdout=sp.PIPE)
    streamdata = child.communicate()[0]
    rc = child.returncode
    #print(streamdata.decode())
    if(rc != 0):
        sys.exit()

if __name__ == "__main__":
    starttotal = time.time()
    print('Python', python_version())   
    dimx = 117
    dimy = 86
    dimz = 38
    dstpath = "output_simulation"
    
    ### Configure these:
    of_tss = 10
    grain = 0.01
    srcpath = 'CASE_3.5_0.5_par'
    last_ts = 7.02
    PARALLEL = 16
    print("----------------------------------------------------------------------")

    createDirTree(srcpath, dstpath)
    first_ts = get_first_ts(srcpath)
    print("----------------------------------------------------------------------")
    end = time.time()
    print("@@@TIMING - Init: ", end - starttotal)
    start = time.time()
    print("----------------------------------------------------------------------")

    ts_curr = "{:.2f}".format(round(float(first_ts) + grain, 2))
    
    ts_end = "{:.2f}".format(last_ts)
    with open(dstpath + "/params", "w") as paramsfile:
        paramsfile.write("ts_start\t%s;\nts_end\t%s;\n" % (ts_curr, ts_end))
    #command = "pimpleFoam -case %s" % dstpath

    command = "decomposePar -case %s -force" % (dstpath)
    print("%s" % (command))
    child = sp.Popen(command.split(), stdout=sp.PIPE)
    streamdata = child.communicate()[0]
    command = "mpirun -n %d pimpleFoam -case %s -parallel" % (PARALLEL, dstpath)
    
    print("\n$ %s (from timestep %s to %s)" % (command, ts_curr, ts_end))
    child = sp.Popen(command.split(), stdout=sp.PIPE)
    streamdata = child.communicate()[0]
    rc = child.returncode
    if(rc != 0):
        sys.exit()
    #print(streamdata.decode())
              
    command = "./parReconstructPar.sh -c %s -n %d" % (dstpath, PARALLEL)
    print("%s" % (command))
    child = sp.Popen(command.split(), stdout=sp.PIPE)
    streamdata = child.communicate()[0]
     
    print("----------------------------------------------------------------------")
    endtotal = time.time()
    print("@@@TIMING - Total: ", endtotal - starttotal)
    #with open(dstpath + "/log", "w") as file:
    #    file.write(streamdata.decode())
    print("----------------------------------------------------------------------")
