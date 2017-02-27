#include <cstdio>
#include <csignal>
#include <mpi.h>
#include <errno.h>
#include <dpd-rng.h>
#include <map>
#include ".conf.h" /* configuration file (copy from .conf.test.h) */
#include "common.h"
#include "common.tmp.h"
#include "bund.h"

int main(int argc, char **argv) {
  int rank, device = 0;
  int ranks[3] = {1, 1, 1}; /* default `xrank', `yrank', `zrank' */
  for (int iarg = 1; iarg < argc && iarg <= 3; iarg++)
    ranks[iarg - 1] = atoi(argv[iarg]);

  CC(cudaSetDevice(device));
  MC(MPI_Init(&argc, &argv));
  MC(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

  MPI_Comm cartcomm, activecomm = MPI_COMM_WORLD;
  MC(MPI_Cart_create(activecomm, 3, ranks, periods, 0, &cartcomm));
  activecomm = cartcomm;
  MC(MPI_Barrier(activecomm));

  sim::init(cartcomm, activecomm);
  sim::run();
  sim::close();

  if (activecomm != cartcomm) MC(MPI_Comm_free(&activecomm));
  MC(MPI_Comm_free(&cartcomm));
  MC(MPI_Finalize());
}
