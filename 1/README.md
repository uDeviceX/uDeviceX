# status

`ssh daint sh /scratch/snx1600/lisergey/check`

# git: 866961d

    :  3.6e+02  1.1e+00 [ 1.8e+05 -2.2e+01 -6.7e+01]  1.0e+01
    DBG: x = 21.3908 (L = 33)
    ../src/sim/_dang/step.h:23: (sub::check_pos_pu): INVALID op
    wild particle: [21.3908 -1.81813 2.60414]
    ../src/odstr/halo/_dang/check.h:22: void dev::check(const float *): block: [432,0,0], thread: [127,0,0] Assertion `0` failed.
    : ../src/odstr/imp.cu:61: device-side assert triggered
    srun: error: nid02611: task 0: Exited with exit code 1
    srun: Terminating job step 3411379.0


    15     update_solvent();
    16     O;
    17     if (solids0) update_solid();
    18     if (rbcs)    update_rbc();
    19     O;
    20     if (wall0) bounce();
    21     O;
    22     if (sbounce_back && solids0) bounce_solid(it);
    23     O;
    24 }
    25
    26 void step(float driving_force0, bool wall0, int it) {
    27     odstr();
    28     step0(driving_force0, wall0, it);
    29 }


from solid_diag_000.txt

    +3.550000e+02 +2.252780e+01 +1.610570e+01 +1.576078e+01 +3.012308e+00 +2.130802e-03 -1.586296e-02
