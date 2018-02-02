#include <stdio.h>

#include "utils/msg.h"
#include "mpi/glb.h"
#include "scheme/time/imp.h"

int main(int argc, char **argv) {
    float ts, dt, dump;
    int i;
    m::ini(&argc, &argv);
    msg_ini(m::rank);
    Time *time;
    ts = 0; dt = 0.1; dump = 0.5;
    time_ini(ts, /**/ &time);

    for (i = 0; i < 100; i++) {
        time_step(time, dt);
        printf("%g %d\n", time_current(time), time_cross(time, dump));
    }
    time_fin(time);
    m::fin();
}
