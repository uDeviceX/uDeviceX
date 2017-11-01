#!/bin/bash -l
#
#SBATCH --job-name=chsolL1D15
#SBATCH --time=24:00:00
#SBATCH --ntasks=3
#SBATCH --ntasks-per-node=1
#SBATCH --output=slurm_out.%j.o
#SBATCH --constraint=gpu
#SBATCH --error=slurm_out.%j.e
#SBATCH --account=ch7

# ======START=====
module load daint-gpu
module load slurm
export CRAY_CUDA_MPS=1

# Run simulation
srun -u -n 3 ./udx 3 1 1 

# =====END====
