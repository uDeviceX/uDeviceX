#include <mpi.h>
#include <stdio.h>

#include "utils/msg.h"
#include "utils/mc.h"
#include "mpi/glb.h"
#include "mpi/wrapper.h"
#include "scheme/time/imp.h"

int main(int argc, char **argv) {
    float s, ts, dt, dump;
    int i, rank, dims[3];
    Time *t;
    MPI_Comm cart;
    
    m::ini(&argc, &argv);
    m::get_dims(&argc, &argv, dims);
    m::get_cart(MPI_COMM_WORLD, dims, &cart);

    MC(m::Comm_rank(cart, &rank));
    msg_ini(rank);
    ts = 0; dt = 0.01; dump = 0.25; s = 0.0001;
    time_ini(ts, /**/ &t);
    for (i = 0; i < 1000; i++) {
        time_next(t, dt += s);
        if (time_cross(t, dump))
            printf("%d %ld %g %g %g\n", i, time_iteration(t), time_current(t), time_dt(t), time_dt0(t));
    }
    time_fin(t);

    MC(m::Barrier(cart));
    m::fin();
}
