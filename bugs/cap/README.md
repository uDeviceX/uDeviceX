# msg

`/scratch/snx3000/lisergey/bug/0`

    : mpi size: 1
    : will take 9999999000 steps
    : recolor
    : ../src/comm/imp/main.h: 19: Error: sending more than capacity in fragment 24 : (11952 / 1000);  called from:
    : ../src/exch/mesh/imp/com.h: 6
	
    udx: ../src/utils/error.cpp:33: void UdxError::report(const char*, int): Assertion `0' failed.
    srun: error: nid05586: task 0: Aborted
    srun: Terminating job step 4138641.0
