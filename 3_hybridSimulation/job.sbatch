#!/bin/bash
#SBATCH --ntasks 16
#SBATCH --time 00-06:00:00
#SBATCH --exclusive
#SBATCH --job-name="h_0.5"
#SBATCH -o "%x-%j.out"

#python run_workflow2.py
python run_simulation.py
#python run_simulation_par.py
