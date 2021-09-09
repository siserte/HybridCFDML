#!/usr/bin/env python
# coding: utf-8
import os
import sys
import shutil  
import subprocess as sp
from platform import python_version
from multiprocessing import Pool, cpu_count
import numpy as np

RESET = True
PATH = "/home/siserte/ainaBuildingsV4/"
BASECASEDIR = PATH + '/CASE_base/'
EXPDIR = PATH + '/experiments/'
NPZDIR = PATH + '/NPZs/'
#CASES = [3, 3.7, 4, 4.5, 5, 6]
#CASES = np.array(range(300,401,10))/100
CASES = np.array(range(400,500,10))/100
#CASES = np.array(range(500,601,10))/100
CPUS = 4
  
def writeParamsFile(path, uref):
    filename = path + "/params"
    print("Writing file %s ..." % filename)
    with open(filename, 'w') as f:
        f.write("uref\t%s;\n" % uref)
    command = "touch %s/CASE_%s.foam" % (path, str(uref))
    sp.run(command.split(), capture_output=True)
    
def createNewExperiment(dstdir):
    print("Copying %s in %s ..." % (BASECASEDIR, dstdir))
    shutil.copytree(BASECASEDIR, dstdir, dirs_exist_ok=True)

def runExperiment(dstdir):
    caseid = dstdir.split("/")[-1].split("_")[-1]
    filename = dstdir + "/mypimple.sbatch"
    print("Writing file %s ..." % filename)
    with open(filename, 'a') as f:
        f.write("#!/bin/bash\n")
        f.write("#SBATCH --job-name=\"c%s\"\n" % caseid)
        f.write("#SBATCH -o %s/slurm-\"%%x-%%j.out\"\n" % dstdir)
        f.write("#SBATCH --ntasks %d\n" % (CPUS))                       
        f.write("cd %s\n" % (dstdir))
        f.write("my_pimpleFoam > %s/log\n" % (dstdir))
        c = dstdir.split("/")[-1].split("_")[-1]
        f.write("python %s/2-generate_dataset.py %s/CASE_%s case%s %d\n" % (PATH, EXPDIR, c, c, CPUS))

    command = "sbatch %s/mypimple.sbatch" % (dstdir)
    #sp.run(command.split(), capture_output=True)
    print(command)
    child = sp.Popen(command.split(), stdout=sp.PIPE)
    streamdata = child.communicate()[0]
    #rc = child.returncode  
    
if __name__ == "__main__":
    print('Python', python_version())   

    if not os.path.exists(EXPDIR):
        #print("Reset directory %s ..." % (EXPDIR))
        #shutil.rmtree(EXPDIR)
        os.mkdir(EXPDIR)
        
    if not os.path.exists(NPZDIR):
        #print("Reset directory %s ..." % (NPZDIR))
        #shutil.rmtree(NPZDIR)
        os.mkdir(NPZDIR)
    
    for case in CASES:
        dstdir = '%s/CASE_%s' % (EXPDIR, str(case))
        createNewExperiment(dstdir)
        writeParamsFile(dstdir, str(case))
        runExperiment(dstdir)
