#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#include <conf.h>
#include "inc/conf.h"

#include "mpi/wrapper.h"
#include "inc/conf.h"
#include "utils/mc.h"
#include "mpi/glb.h"

namespace m { /* MPI */
static const int D = 3;
static int periods[D] = {true, true, true};
static const bool reorder = false;
int rank, size, dims[D];

static void shift(int *argc, char ***argv) {
    (*argc)--;
    (*argv)++;
}
static int eq(const char *a, const char *b) { return strcmp(a, b) == 0; }

static void set_dims(int *argc, char ***argv) {
    int i, d, ac;
    char **av;

    // defaults
    dims[0] = dims[1] = dims[2] = 1;
    ac = *argc; av = *argv;

    // skip executable
    shift(&ac, &av);

    for (i = 0; ac > 0 && i <= 3; i++) {
        if (eq(av[0], "--")) {
            shift(&ac, &av);
            break;
        }
        if ( (d = atoi(av[0])) == 0 ) break;
        dims[i] = d;
        shift(&ac, &av);
    }

    *argc = ac; *argv = av;
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
    MC(m::Cart_create(MPI_COMM_WORLD, D, dims, periods, reorder,   &m::cart));
}

void fin() {
    MC(m::Barrier(m::cart));
    MC(m::Finalize());
}
}
