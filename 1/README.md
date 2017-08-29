# git: 89cfd30, _release
## n = 1
`daint:/scratch/snx1600/lisergey/udx/gen/n1`

    1.9e+03  1.1e+00 [ 1.8e+05 -7.5e+01  6.3e+01]  1.0e+01
    2.0e+03  1.1e+00 [ 1.8e+05  4.9e+01  4.2e+01]  1.0e+01
    ../src/sim/_release/0dev/distr.h:2: device-side assert triggered

1/.000:

## n = 40
`daint:/scratch/snx1600/lisergey/udx/gen/n40`

    :  3.4e+03  1.6e+00 [ 8.5e+06 -6.4e+04 -1.6e+03]  2.0e+01
    :  3.4e+03  1.6e+00 [ 8.5e+06 -6.4e+04 -1.8e+03]  2.0e+01
    srun: error: nid06556: task 37: Exited with exit code 1
    srun: Terminating job step 3340744.0
	
1/.037:
	
    sim.impl: 0/0 Solids survived
    ../src/odstr/imp.cu:61: misaligned address
	
slurm-*.out:

    ../src/odstr/dev/check.h:10: void odstr::sub::dev::check_cel(float, int, int): block: [0,0,0], thread: [30,0,0] Assertion `0` failed.
     odstr: i = 32 (L = 32) from x = 16
     odstr: i = 32 (L = 32) from x = 16
	 ...

# git: 89cfd30, _safe
## n = 1
`daint:/scratch/snx1600/lisergey/safe/n40`


## n = 40

`daint:/scratch/snx1600/lisergey/safe/n40`
