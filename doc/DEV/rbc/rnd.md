# random force

Activate by `RBC_RND` in `conf.h`. If envariment variable `RBC_RNC` is
set it is used as seed. Magnitude of the force is

    g = RBCgammaC; T = RBCkbT
    f0  = sqrtf(2*g*T/dt)*rnd

