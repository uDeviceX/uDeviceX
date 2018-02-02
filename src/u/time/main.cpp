#include <stdio.h>

#include "utils/msg.h"
#include "mpi/glb.h"
#include "scheme/time/imp.h"

int main(int argc, char **argv) {
    float dt0;
    m::ini(&argc, &argv);
    msg_ini(m::rank);
    Time *t;
    dt0 = 0.1;
    time_ini(dt0, /**/ &t);

    time_fin(t);
    m::fin();
}
