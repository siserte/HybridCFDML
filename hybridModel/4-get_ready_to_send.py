#!/usr/bin/env python
# coding: utf-8
import os
import sys
import numpy as np
import glob
import importlib
import pathlib
from joblib import dump, load
from platform import python_version
import subprocess as sp


def is_number(string):
    try:
        float(string)
        return True
    except ValueError:
        return False
   
if __name__ == "__main__":
    error = False
    n_args = len(sys.argv)
    path = sys.argv[1]
    #python 4-get_ready_to_send.py output_dataset
    
    print("Removing files in timesteps of %s..." % (path))
    (dirpath, dirnames, filenames) = next(os.walk(path))
    for dir in dirnames:
        if (is_number(dir)):
            num = float(dir)
            if (num > 1):
                command = "rm -rf %s/%s/k %s/%s/nut %s/%s/omega %s/%s/p %s/%s/phi %s/%s/uniform %s/%s/include" % (path, dir,path, dir,path, dir,path, dir,path, dir,path, dir,path, dir)
                #print(command)
                child = sp.Popen(command.split(), stdout=sp.PIPE)
                streamdata = child.communicate()[0]

    command = "tar -czvf %s.tar.gz %s" % (path, path)
    print(command)
    child = sp.Popen(command.split(), stdout=sp.PIPE)
    streamdata = child.communicate()[0]
    