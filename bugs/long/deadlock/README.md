# stdout

    8.050000e+01    6.9986699416e+01        1.4244249395e+06        -1.6932099994e-03       1.6697941122e-02
    srun: error: nid03014: task 0: Segmentation fault
    srun: Terminating job step 2945305.1

# deadlock

Two times

    2.250000e+01    1.6080401056e-02        1.1177369989e+02        -8.0807306522e-03       -2.4729833221e-02

# gdb
git: ae0ba2e

(gdb) bt
#0  0x00007ffff71e9eea in MPIDI_CH3I_Progress ()
   from /usr/lib64/mpich/lib/libmpich.so.10
#1  0x00007ffff729668a in PMPI_Recv () from /usr/lib64/mpich/lib/libmpich.so.10
#2  0x000000000042881b in rex::recvP2 (cart=-2080374781,
    ranks=ranks@entry=0x5726824 <x::tc+4>, tags=tags@entry=0x57267a0 <x::tr>,
    t=...) at ../../../src/rex/recv.h:49
#3  0x0000000000429d9d in rex0 (nw=1, w=...) at ../../../src/x/impl.h:36
#4  x::rex (w=std::vector of length 1, capacity 1 = {...})
    at ../../../src/x/impl.h:55
#5  0x000000000042a207 in sim::forces (wall0=wall0@entry=false)
    at ../../../src/sim/force1.h:17
#6  0x000000000042a356 in sim::step0 (driving_force0=0.00999999978,
    wall0=<optimized out>, it=2834) at ../../../src/sim/step.h:4
#7  0x000000000042a4dd in step (it=<optimized out>, wall0=false,
    driving_force0=0.00999999978) at ../../../src/sim/step.h:16
#8  sim::run (ts=ts@entry=1000, te=te@entry=2000000)
    at ../../../src/sim/run.h:11
#9  0x000000000042a67d in sim::sim_gen () at ../../../src/sim/imp.h:64
#10 0x00000000004033fa in main (argc=<optimized out>, argv=<optimized out>)
    at ../../../src/main.cu:21
