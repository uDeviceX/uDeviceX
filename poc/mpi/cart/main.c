#include <stdio.h>
#include <mpi.h>

#define ndims 3
int rank, coords[ndims];
int    dims[ndims] = {2, 2, 2};
int periods[ndims] = {0, 0, 0};
int reorder = 0;
MPI_Comm cart;

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, reorder,   &cart);

  MPI_Comm_rank(cart,   &rank);
  MPI_Cart_coords(cart, rank, ndims,   coords);
  printf("N, rank, coords[3]: %d %d   %d %d %d\n", N, rank, coords[0], coords[1], coords[2]);
  
  MPI_Finalize();
  return 0;
}
