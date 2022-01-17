# HybridCFDML

Results presented in this section have been obtained leveraging the following hardware:
  - DLM training was performed on the cluster CTE-Power at Barcelona Supercomputing Center (BSC). CTE-Power nodes are equipped each with two processors IBM Power9 8335-GTH @ 2.4GHz with a total of 160 threads, 512GB of main memory, and four GPU NVIDIA V100 with 16GB HBM2.
  - Simulations and DLM predictions were executed on Tirant III supercomputer at Universitat de València (UV).
    Tirant cluster is not GPU-enabled, so each server is composed of two sockets Intel Xeon SandyBridge E5-2670 @ 2.6GHz with a total of 16 threads, and 32GB of main memory.

In addition, regarding the software stack, CFD simulations have been performed in Tirant III with OpenFOAM v2006.
While the DLM has been trained on CTE-Power with Python 3.7.4, NumPy 1.18.4, Scikit-learn 0.23.1, and Keras 2.4 over Tensorflow 2.3.
The code and datasets are available in https://github.com/siserte/HybridCFDML.

Dataset is composed of 31 cases with Uref profiles going from 3 to 6 in 0.1 m/s.

Test subdataset is composed of 7 cases corresponding to the following Uref profile velocities: 3.5, 5.8, 4.7, 3.3, 5.5, 5.3, 4.9 m/s

The next example compares the same simulation executed with a traditional CFD solver (top), and the solution provided by the hybrid model (bottom):

<img src="https://github.com/siserte/HybridCFDML/blob/main/cfd.gif" width="600" />
