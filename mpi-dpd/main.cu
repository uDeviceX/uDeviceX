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

  const bool mpi_thread_safe = argp("-mpi_thread_safe").asBool(true);

  CC(cudaSetDevice(0));
  CC(cudaDeviceReset());
  int rank;
  if (mpi_thread_safe) {
    // needed for the asynchronous data dumps
    setenv("MPICH_MAX_THREAD_SAFETY", "multiple", 0);
    int provided_safety_level;
    MC(MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE,
			      &provided_safety_level));
    MC(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

    if (provided_safety_level != MPI_THREAD_MULTIPLE) {
      if (rank == 0)
	printf(
	    "ooooooooops MPI thread safety level is just %d. Aborting now.\n",
	    provided_safety_level);
      abort();
    } else if (rank == 0)
      printf("I have set MPICH_MAX_THREAD_SAFETY=multiple\n");
  } else {
    MC(MPI_Init(&argc, &argv));
    MC(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    const char *env_thread_safety = getenv("MPICH_MAX_THREAD_SAFETY");
    if (rank == 0 && env_thread_safety)
      printf("I read MPICH_MAX_THREAD_SAFETY=%s", env_thread_safety);
  }

  MPI_Comm activecomm = MPI_COMM_WORLD;

  bool reordering = true;
  const char *env_reorder = getenv("MPICH_RANK_REORDER_METHOD");
  // reordering of the ranks according to the computational domain and
  // environment variables
  if (atoi(env_reorder ? env_reorder : "-1") == atoi("3")) {
    reordering = false;
    const bool usefulrank = rank < ranks[0] * ranks[1] * ranks[2];
    MC(MPI_Comm_split(MPI_COMM_WORLD, usefulrank, rank, &activecomm));
    MC(MPI_Barrier(activecomm));
    if (!usefulrank) {
      printf("rank %d has been thrown away\n", rank);
      fflush(stdout);
      MC(MPI_Barrier(activecomm));
      MPI_Finalize();
      return 0;
    }

    MC(MPI_Barrier(activecomm));
  }

  MPI_Comm cartcomm;
  int periods[] = {1, 1, 1};
  MC(MPI_Cart_create(activecomm, 3, ranks, periods, (int)reordering,
			    &cartcomm));
  activecomm = cartcomm;
  MC(MPI_Barrier(activecomm));
  if (rank == 0) argp.print_arguments();
  MC(MPI_Barrier(activecomm));

  sim::sim_init(cartcomm, activecomm);
  sim::sim_run();
  sim::sim_close();

  if (activecomm != cartcomm) MC(MPI_Comm_free(&activecomm));
  MC(MPI_Comm_free(&cartcomm));
  MC(MPI_Finalize());
  CC(cudaDeviceSynchronize());
  CC(cudaDeviceReset());
}
