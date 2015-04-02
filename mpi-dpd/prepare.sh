#/usr/local/bin/bash

module swap PrgEnv-cray PrgEnv-gnu
module load cudatoolkit
module load cray-hdf5-parallel
export LD_LIBRARY_PATH=${PWD}/../cuda-dpd-sem/dpd:${PWD}/../cuda-rbc/:${PWD}/../cuda-ctc/:$LD_LIBRARY_PATH
