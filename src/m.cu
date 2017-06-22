#include <mpi.h>
#include "m.h"

namespace m { /* MPI (man MPI_Cart_get) */
const int d = 3;
int periods[d] = {true, true, true};
/* set in main */
int rank, size, coords[d], dims[d];
const bool reorder = false;
MPI_Comm cart;
}
