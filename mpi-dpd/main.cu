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
  if (argc < 4) {
    printf("usage: ./mpi-dpd <xranks> <yranks> <zranks>\n");
    exit(-1);
  }

  int rank, ranks[3];
  for (int i = 0; i < 3; ++i) ranks[i] = atoi(argv[1 + i]);

  int device = 0;
  CC(cudaSetDevice(device));

  MC(MPI_Init(&argc, &argv));
  MC(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

  MPI_Comm cartcomm, activecomm = MPI_COMM_WORLD;
  int periods[] = {1, 1, 1};
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
