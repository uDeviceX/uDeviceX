#include <mpi.h>
#include "m.h"

namespace m { /* MPI (man MPI_Cart_get) */
  const int d = 3;
  const bool reorder = false;
  int periods[3] = {true, true, true};
  /* set in main */
  int  rank, coords[3], dims[3];

  extern MPI_Comm cart;
}
