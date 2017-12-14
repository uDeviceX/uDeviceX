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

static void set_dims(int *argc, char ***argv) {
    int i, d;
    char **av;
    int ac;

    ac = *argc;
    av = *argv;
    
    dims[0] = dims[1] = dims[2] = 1;

    d = 1;
    for (i = 1; d > 0 && i < ac && i <= 3; i++) {
        d = atoi(av[i]);
        if (d > 0) dims[i - 1] = d;
    }

    ac -= i - 1;
    av += i - 1;

    *argc = ac;
    *argv = av;
}
void ini(int *argc, char ***argv) {
    if (m::Init(argc, argv) != MPI_SUCCESS) {
        fprintf(stderr, ": m::Init failed\n");
        exit(2);
    }
    if (m::Errhandler_set(MPI_COMM_WORLD, MPI_ERRORS_RETURN) != MPI_SUCCESS) {
        fprintf(stderr, ": m::Errhandler_set\n");
        exit(2);
    }

    set_dims(argc, argv);
    
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
