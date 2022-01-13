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
    dstpath = "output_dataset_200"
    npzfile = "tmp_dataset"
    n_input = 3
    
    ### Configure these:
    n_output = 10
    of_tss = 10
    of_tss_ini = 3
    #model_path = "../surrogateModelContinuous/MODCHANFULL10-0-100"
    model_path = "MODCHAN10-0-200"
    grain = 0.01
    DRYRUN = False       #Copy ts from an OF simulation instead of simulating them.
    OFFIRST = True    #Start the execution simulating instead of predicting.
    CONTINUE = False     #Continue a previous execution instead of start from scratch.
    BREAKFIRST = False   #Run a single iteration: simulate (if OFFIRST), generate, predict, and regenerate.
    srcpath = 'CASE_test_3.5_0.5_lite'
    last_ts = 7.01

    print("----------------------------------------------------------------------")
    if CONTINUE:
       srcpath = dstpath
       OFFIRST = False
    else:
        createDirTree(srcpath, dstpath)
    first_ts = get_first_ts(srcpath)
    print("----------------------------------------------------------------------")
    end = time.time()
    print("@@@TimeInit: ", end - starttotal)
    start = time.time()
    print("----------------------------------------------------------------------")

    ts_curr = "{:.2f}".format(round(float(first_ts) + grain, 2))
    last_of = "{:.2f}".format(round(float(first_ts), 2))
    
    while float(ts_curr) < last_ts:
    
        if OFFIRST:
            ts_end = "{:.2f}".format(float(ts_curr) + ((of_tss_ini-2) * grain))
            with open(dstpath + "/params", "w") as paramsfile:
                paramsfile.write("ts_start\t%s;\nts_end\t%s;\n" % (ts_curr, ts_end))
            command = "pimpleFoam -case %s" % dstpath
            print("\n$ %s (from timestep %s to %s)" % (command, ts_curr, ts_end))
            child = sp.Popen(command.split(), stdout=sp.PIPE)
            streamdata = child.communicate()[0]
            rc = child.returncode
            if(rc != 0):
                sys.exit()
            #print(streamdata.decode())
            
            ts_curr = "{:.2f}".format(round(float(ts_end) + grain, 2))
            last_of = ts_end
            OFFIRST = False
            
            print("----------------------------------------------------------------------")     
            end = time.time()
            print("@@@TimeSimulation: ", end - start, "ts next: ", ts_curr)
            start = time.time()
            print("----------------------------------------------------------------------")
    
        command = "python 1-generate_dataset.py %s %s" % (dstpath, npzfile)
        print("\n$ %s\n" % (command))
        child = sp.Popen(command.split(), stdout=sp.PIPE)
        streamdata = child.communicate()[0]
        rc = child.returncode
        print(streamdata.decode())
        if(rc != 0):
            sys.exit()

        #break  
        
        print("----------------------------------------------------------------------")     
        end = time.time()
        print("@@@TimeDataset: ", end - start, "ts next: ", ts_curr)
        start = time.time()
        print("----------------------------------------------------------------------")

        ts_end = "{:.2f}".format(round(float(ts_curr) + n_output * grain, 2))

        command = "python 2-CNN_prediction.py %s.npz %s %s %s" % (npzfile, model_path, ts_curr, ts_end)
        print("\n$ %s\n" % (command))
        child = sp.Popen(command.split(), stdout=sp.PIPE)
        streamdata = child.communicate()[0]
        rc = child.returncode
        print(streamdata.decode())
        if(rc != 0):
            sys.exit()

        print("----------------------------------------------------------------------")
        end = time.time()
        print("@@@TimePrediction: ", end - start, "s. TS next: ", ts_curr)
        start = time.time()
        print("----------------------------------------------------------------------")

        tslist = []
        aux_ts = int(round(float(ts_curr) * 100))
        for i in range(n_output):
            #print(ts_curr, float(ts_curr), float(ts_curr) * 100, round(float(ts_curr) * 100), aux_ts, i)
            tslist.append(aux_ts + i)
        print(tslist)
        
        last_of_str = parse_OF_dirname(round(float(last_of), 2))        
        p = Pool(min(n_output, cpu_count()))
        #p.map(callRegenerate, range(ts_init, ts_init + n_output))
        p.map(callRegenerate, tslist)
        p.close()
        p.join()

        ts_curr = ts_end
        
        print("----------------------------------------------------------------------")
        end = time.time()
        print("@@@TimeRegenerate: ",  end - start, "s. TS next: ", ts_curr)
        start = time.time()
        print("----------------------------------------------------------------------")
        
        ts_end = "{:.2f}".format(float(ts_curr) + ((of_tss-1) * grain))
        
        if DRYRUN:
            aux_ts = int(float(ts_curr) * 100)
            for i in range(n_output):
                copyts = parse_OF_dirname(float(aux_ts + i) / 100)        
                command = "cp -r CASE_3.5_myPimple/%s %s/" % (copyts, dstpath)
                print("$ %s" % (command))
                child = sp.Popen(command.split(), stdout=sp.PIPE)
                streamdata = child.communicate()[0]
                #rc = child.returncode
        else:
            with open(dstpath + "/params", "w") as paramsfile:
                paramsfile.write("ts_start\t%s;\nts_end\t%s;\n" % (ts_curr, ts_end))
            command = "pimpleFoam -case %s" % dstpath
            print("\n$ %s (from timestep %s to %s)" % (command, ts_curr, ts_end))
            child = sp.Popen(command.split(), stdout=sp.PIPE)
            streamdata = child.communicate()[0]
            rc = child.returncode
            if(rc != 0):
                sys.exit()    
                
        last_of = ts_end
        
        print("----------------------------------------------------------------------")
        end = time.time()
        print("@@@TimeSimulation: ", end - start)
        start = time.time()
        print("----------------------------------------------------------------------")

        print("----------------------------------------------------------------------")
        end = time.time()
        ts_curr = "{:.2f}".format(round(float(ts_end), 2))
        print("@@@Time accumulated until ts %s: " % (ts_curr), end - starttotal)
        print("----------------------------------------------------------------------")

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
