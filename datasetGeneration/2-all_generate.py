#!/usr/bin/env python
# coding: utf-8
import os
import sys
import shutil  
import subprocess as sp
from platform import python_version
from multiprocessing import Pool, cpu_count
import numpy as np

if __name__ == "__main__":
    print('Python', python_version())   
    #CASES = np.array(range(300,400,10))/100
    #CASES = np.array(range(401,502,10))/100
    CASES = np.array(range(500,601,10))/100
    PATH = "/home/siserte/ainaBuildingsV4/"
    EXPDIR = PATH + '/experiments/'
    
    for c in CASES:
        #print(c)
        command = "python 2-generate_dataset.py %s/CASE_%s case%s 24" % (EXPDIR, c, c)
        print(command)
        child = sp.Popen(command.split(), stdout=sp.PIPE)
        streamdata = child.communicate()[0]