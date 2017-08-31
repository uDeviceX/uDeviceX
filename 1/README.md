# _dang after Lukas fix

dang/fix/n40

    ../src/odstr/halo/_dang/reg.h:15: 36100 > 12 for fid: 26

# status

`ssh daint sh /scratch/snx1600/lisergey/check`

# nosolid

daint:/scratch/snx1600/lisergey/nsolid/nomem

    :  5.8e+03  1.2e+00 [ 1.9e+05 -5.2e+01 -9.0e+00]  1.1e+01
    : ../src/dpdr/imp.cu:73: an illegal memory access was encountered
    srun: error: nid02701: task 0: Exited with exit code 1
    srun: Terminating job step 3390136.0

# solid, XS=16

daint:/scratch/snx1600/lisergey/small16

    :  3.2e+03  3.5e-02 [ 3.9e+03  8.3e-01 -9.6e+00]  3.2e-01
    : ../src/sim/_dang/0dev/distr.h:2: an illegal memory access was encountered
    srun: error: nid06336: task 0: Exited with exit code 1
    srun: Terminating job step 3392233.0
