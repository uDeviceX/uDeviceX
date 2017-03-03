#!/bin/bash -l
#
#SBATCH --job-name="rbc_shear"
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=rbc_shear.%j.o
#SBATCH --error=rbc_shear.%j.e
#SBATCH --constraint=gpu
#SBATCH --account=ch7

#======START=====

source ~/.bashrc
module load cudatoolkit

export HEX_COMM_FACTOR=2

#srun --ntasks 1 --export ALL ./test 1 1 1 -rbcs -tend=10000 -steps_per_dump=1000 -shrate=1e-1 -RBCx0=0.4 -RBCp=5e-3 -RBCkb=40 -RBCka=4900 -RBCkd=100 -RBCkv=5000 -RBCgammaC=30 -RBCtotArea=124 -RBCtotVolume=90 -RBCfk=0

rm -rf h5 ply
srun --ntasks 1 --export ALL ./test 1 1 1

#=====END====
