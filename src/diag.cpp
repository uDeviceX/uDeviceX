#include <stdio.h>
#include <mpi.h>
#include <conf.h>
#include "inc/conf.h"

#include "l/m.h"
#include "m.h"
#include "inc/type.h"
#include "inc/mpi.h"
#include "diag.h"

static float sq(float x) { return x*x; }

int reduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype,
           MPI_Op op, int root, MPI_Comm comm) {
    return l::m::Reduce(sendbuf, recvbuf, count, datatype, op, root, comm);
}

void diagnostics(Particle *pp, int n, int id) {
    int i, c;
    double ke, kbt;
    FILE * f;
    double p[] = {0, 0, 0};
    for (i = 0; i < n; ++i) for (c = 0; c < 3; ++c) p[c] += pp[i].v[c];

    MC(l::m::Reduce(m::rank == 0 ? MPI_IN_PLACE : &p,
                    m::rank == 0 ? &p : NULL, 3,
                    MPI_DOUBLE, MPI_SUM, 0, l::m::cart) );
    ke = 0;
    for (i = 0; i < n; ++i)
        ke += sq(pp[i].v[0]) + sq(pp[i].v[1]) + sq(pp[i].v[2]);

    MC(l::m::Reduce(m::rank == 0 ? MPI_IN_PLACE : &ke,
                    &ke,
                    1, MPI_DOUBLE, MPI_SUM, 0, l::m::cart));
    MC(l::m::Reduce(m::rank == 0 ? MPI_IN_PLACE : &n,
                    &n, 1, MPI_INT, MPI_SUM, 0, l::m::cart));

    if (m::rank == 0) {
        kbt = 0.5 * ke / (n * 3. / 2);
        static bool firsttime = true;
        f = fopen(DUMP_BASE "/diag.txt", firsttime ? "w" : "a");
        firsttime = false;
        if (id == 0) fprintf(f, "# TSTEP\tKBT\tPX\tPY\tPZ\n");
        fprintf(stderr, "%.3e %.3e %.3e %.3e %.3e\n", id * dt, kbt, p[0], p[1], p[2]);
        fprintf(f, "%e\t%.10e\t%.10e\t%.10e\t%.10e\n", id * dt, kbt, p[0], p[1], p[2]);
        fclose(f);
    }
}
