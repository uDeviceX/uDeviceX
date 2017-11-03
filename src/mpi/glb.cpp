#include <stdlib.h>
#include <mpi.h>

#include <conf.h>
#include "inc/conf.h"

#include "mpi/wrapper.h"
#include "inc/conf.h"
#include "utils/mc.h"
#include "mpi/glb.h"

namespace m { /* MPI */
static const int d = 3;
static int periods[d] = {true, true, true};
static const bool reorder = false;
int rank, size, coords[d], dims[d];

int lx() { /* domain sizes */
    enum {X};
    return XS * dims[X];
}
int ly() {
    enum {X, Y};
    return YS * dims[Y];
}
int lz() {
    enum {X, Y, Z};
    return ZS * dims[Z];
}

float x2g(float r) { /* local to global */
    enum {X};
    return (m::coords[X] + 0.5)*XS  + r;
}
float y2g(float r) { /* local to global */
    enum {X, Y};
    return (m::coords[Y] + 0.5)*YS  + r;
}
float z2g(float r) { /* local to global */
    enum {X, Y, Z};
    return (m::coords[Z] + 0.5)*ZS  + r;
}

static void set_dims(int argc, char **argv) {
    int i;
    dims[0] = dims[1] = dims[2] = 1;
    for (i = 1; i < argc && i <= 3; i++) dims[i - 1] = atoi(argv[i]);
}
void ini(int argc, char **argv) {
    set_dims(argc, argv);

    if (m::Init(&argc, &argv) != MPI_SUCCESS) {
        fprintf(stderr, ": m::Init failed\n");
        exit(2);
    }

    m::Errhandler_set(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
    MC(m::Comm_rank(MPI_COMM_WORLD,   &rank));
    MC(m::Comm_size(MPI_COMM_WORLD,   &size));
    MC(m::Cart_create(MPI_COMM_WORLD, d, dims, periods, reorder,   &m::cart));
    MC(m::Cart_coords(m::cart, rank, d,   coords));
}

void fin() {
    MC(m::Barrier(m::cart));
    MC(m::Finalize());
}
}
