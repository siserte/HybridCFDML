#!/bin/bash
#SBATCH --ntasks 40
#SBATCH --gres=gpu:1 
##SBATCH --time 00-06:00:00
##SBATCH --exclusive
#SBATCH --job-name="training"
#SBATCH -o "%x-%j.out"

python 2-Train-per-layers.py

