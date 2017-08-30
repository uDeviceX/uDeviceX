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

# git: 52f3b20, _release

## n = 1
	`slurm-3354480.out:`

	:  1.9e+03  1.1e+00 [ 1.8e+05 -2.7e+01  2.2e+01]  1.0e+01
	:  1.9e+03  1.1e+00 [ 1.8e+05 -8.6e+00  4.0e+01]  1.0e+01
	:  1.9e+03  1.1e+00 [ 1.8e+05  1.6e+01 -3.4e+01]  1.0e+01
	: ../src/sim/_release/0dev/distr.h:2: an illegal memory access was encountered

# git: ac7afcf, _release

  union means without union.

## n = 1
	union/n1

	:  1.5e+03  1.1e+00 [ 1.8e+05  8.6e+00 -4.0e+01]  1.0e+01
	: ../src/sim/_release/0dev/distr.h:2: an illegal memory access was encountered

## n = 40
	union/n40, runs for 10h

# git: ac7afcf, _release, no solid

## n = 1
	nosolid/n1, runs for 10h

## n = 40
	nosolid/n40, runs for 10h

# git: e8ab233, _release

   add dSync in front of distr_solid

## n = 1
	dsync/n1

	:  1.9e+03  1.1e+00 [ 1.8e+05 -4.7e+01  3.9e+01]  1.1e+01
	: ../src/odstr/imp.cu:61: an illegal memory access was encountered

# git: 29b6fc3, _dang

dang: checks `iidx` buffer overflow
ODSTR_FACTOR = 3

	 :  1.0e+03  1.1e+00 [ 1.8e+05 -5.6e+01  2.1e+01]  1.0e+01
	 : ../src/sim/_dang/0dev/distr.h:2: an illegal memory access was encountered
