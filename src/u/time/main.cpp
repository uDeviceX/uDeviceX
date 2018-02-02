#include <stdio.h>

#include "utils/msg.h"
#include "mpi/glb.h"
#include "scheme/time/imp.h"

int main(int argc, char **argv) {
    float ts;
    m::ini(&argc, &argv);
    msg_ini(m::rank);
    Time *t;
    ts = 0;
    time_ini(ts, /**/ &t);

    time_fin(t);
    m::fin();
}
