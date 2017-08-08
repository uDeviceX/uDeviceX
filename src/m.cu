#include <mpi.h>
#include "m.h"

namespace m { /* MPI */
const int d = 3;
int periods[d] = {true, true, true};
int rank, size, coords[d], dims[d];
const bool reorder = false;
}
