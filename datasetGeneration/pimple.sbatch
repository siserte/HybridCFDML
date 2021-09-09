#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks 12 //cores

#decomposePar

#mpirun -np $SLURM_NTASKS --host $SLURM_NODELIST --oversubscribe simpleFoam >log -parallel
#my_pimpleFoam >log
pimpleFoam >log
#reconstructPar
