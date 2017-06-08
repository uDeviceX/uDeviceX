ini() { module load cray-hdf5-parallel cudatoolkit daint-gpu GSL Octave; }

run () { srun ./udx 4 4 4; }
