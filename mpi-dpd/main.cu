#include <cstdio>
#include <csignal>
#include <mpi.h>
#include <errno.h>
#include <dpd-rng.h>
#include <map>
#include "argument-parser.h"
#include ".conf.h" /* configuration file (copy from .conf.test.h) */
#include "common.h"
#include "simulation.h"

float tend;
bool walls, pushtheflow, doublepoiseuille, rbcs, hdf5field_dumps,
    hdf5part_dumps, contactforces;
int steps_per_dump, steps_per_hdf5dump, wall_creation_stepid;

float RBCx0, RBCp, RBCcq, RBCkb, RBCka, RBCkv, RBCgammaC, RBCkd, RBCtotArea,
    RBCtotVolume, RBCscale;

int main(int argc, char **argv) {
  int ranks[3];

  // parsing of the positional arguments
  if (argc < 4) {
    printf("usage: ./mpi-dpd <xranks> <yranks> <zranks>\n");
    exit(-1);
  } else
    for (int i = 0; i < 3; ++i)
      ranks[i] = atoi(argv[1 + i]);

  ArgumentParser argp(vector<string>(argv + 4, argv + argc));

  contactforces = argp("-contactforces").asBool(false);
  doublepoiseuille = argp("-doublepoiseuille").asBool(false);
  hdf5field_dumps = argp("-hdf5field_dumps").asBool(false);
  hdf5part_dumps = argp("-hdf5part_dumps").asBool(false);
  pushtheflow = argp("-pushtheflow").asBool(false);
  rbcs = argp("-rbcs").asBool(false);
  steps_per_dump = argp("-steps_per_dump").asInt(1000);
  steps_per_hdf5dump = argp("-steps_per_hdf5dump").asInt(2000);
  tend = argp("-tend").asDouble(50);
  wall_creation_stepid = argp("-wall_creation_stepid").asInt(5000);
  walls = argp("-walls").asBool(false);

  RBCx0 = argp("-RBCx0").asDouble(0.5);
  RBCp = argp("-RBCp").asDouble(0.0045);
  RBCka = argp("-RBCka").asDouble(4900);
  RBCkb = argp("-RBCkb").asDouble(40);
  RBCkd = argp("-RBCkd").asDouble(100);
  RBCkv = argp("-RBCkv").asDouble(5000);
  RBCgammaC = argp("-RBCgammaC").asDouble(30);
  RBCtotArea = argp("-RBCtotArea").asDouble(124);
  RBCtotVolume = argp("-RBCtotVolume").asDouble(90);

  RBCscale = 1 / rc;

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
  if (rank == 0) argp.print_arguments();
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
