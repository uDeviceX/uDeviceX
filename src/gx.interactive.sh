ini() { module load cray-hdf5-parallel cudatoolkit daint-gpu GSL; }

run () { srun ./udx 4 4 4; }
