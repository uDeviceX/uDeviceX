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
  int ranks[3];

  // parsing of the positional arguments
  if (argc < 4) {
    printf("usage: ./mpi-dpd <xranks> <yranks> <zranks>\n");
    exit(-1);
  } else
    for (int i = 0; i < 3; ++i)
      ranks[i] = atoi(argv[1 + i]);

  CC(cudaSetDevice(0));
  CC(cudaDeviceReset());
  int rank;
  MC(MPI_Init(&argc, &argv));
  MC(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  MPI_Comm activecomm = MPI_COMM_WORLD;
  MPI_Comm cartcomm;
  int periods[] = {1, 1, 1};
  MC(MPI_Cart_create(activecomm, 3, ranks, periods, 0,
			    &cartcomm));
  activecomm = cartcomm;
  MC(MPI_Barrier(activecomm));
  MC(MPI_Barrier(activecomm));

  sim::init(cartcomm, activecomm);
  sim::run();
  sim::close();

  if (activecomm != cartcomm) MC(MPI_Comm_free(&activecomm));
  MC(MPI_Comm_free(&cartcomm));
  MC(MPI_Finalize());
  CC(cudaDeviceSynchronize());
  CC(cudaDeviceReset());
}
