#include <stdio.h>
#include <mpi.h>
#include <conf.h>
#include "inc/conf.h"

#include "utils/msg.h"
#include "mpi/wrapper.h"
#include "mpi/glb.h"
#include "inc/type.h"
#include "utils/error.h"
#include "utils/imp.h"

#include "imp.h"

static float sq(float x) { return x*x; }

static int reduce(MPI_Comm comm, const void *sendbuf0, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op) {
    int root = 0;
    const void *sendbuf = (m::is_master(comm) ? MPI_IN_PLACE : sendbuf0);
    return m::Reduce(sendbuf, recvbuf, count, datatype, op, root, comm);
}

static int sum3(MPI_Comm comm, double *v) {
    return reduce(comm, v, m::is_master(comm) ? v : NULL, 3, MPI_DOUBLE, MPI_SUM);
}

static int sum_d(MPI_Comm comm, double *v) {
    return reduce(comm, v, v, 1, MPI_DOUBLE, MPI_SUM);
}

static int max_d(MPI_Comm comm, double *v) {
    return reduce(comm, v, v, 1, MPI_DOUBLE, MPI_MAX);
}

static int sum_i(MPI_Comm comm, int *v) {
    return reduce(comm, v, v, 1, MPI_INT, MPI_SUM);
}

void diag(MPI_Comm comm, float time, int n, const Particle *pp) {
    if (time < 0) ERR("time < 0");
    enum {X, Y, Z};
    int i, c;
    double k, km, ke; /* particle, total, and maximum kinetic energies */
    double kbt = 0;
    FILE *f = NULL;
    double v[3] = {0};
    for (i = 0; i < n; ++i) for (c = 0; c < 3; ++c) v[c] += pp[i].v[c];
    sum3(comm, v);

    ke = km = 0;
    for (i = 0; i < n; ++i) {
        k = sq(pp[i].v[X]) + sq(pp[i].v[Y]) + sq(pp[i].v[Z]);
        ke += k;
        if (k > km) km = k;
    }

    sum_d(comm, &ke); max_d(comm, &km);
    sum_i(comm, &n);

    if (m::is_master(comm)) {
        if (n) kbt = 0.5 * ke / (n * 3. / 2);
        static bool firsttime = true;
        UC(efopen(DUMP_BASE "/diag.txt", firsttime ? "w" : "a", /**/ &f));
        firsttime = false;
        msg_print("% .1e % .1e [% .1e % .1e % .1e] % .1e", time, kbt, v[X], v[Y], v[Z], km);
        fprintf(f, "%e\t%.10e\t%.10e\t%.10e\t%.10e\t%.10e\n", time, kbt, v[X], v[Y], v[Z], km);
        efclose(f);
    }
}
