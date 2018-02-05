#include <stdio.h>

#include "utils/msg.h"
#include "mpi/glb.h"
#include "scheme/time/imp.h"

int main(int argc, char **argv) {
    float s, ts, dt, dump;
    int i;
    Time *t;
    m::ini(&argc, &argv);
    msg_ini(m::rank);
    ts = 0; dt = 0.01; dump = 0.25; s = 0.0001;
    time_ini(ts, /**/ &t);
    for (i = 0; i < 1000; i++) {
        time_step(t, dt += s);
        if (time_cross(t, dump))
            printf("%d %g %g %g\n", i, time_t(t), time_dt(t), time_dt0(t));
    }
    time_fin(t);
    m::fin();
}
