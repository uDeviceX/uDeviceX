#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include "bop_common.h"
#include "bop_mpi.h"
#include "check.h"

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    
    BopData *d;
    int i, i_, N, n, n_;
    float *data;
    int rank, size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    N = 10;
    n_ = n = N/size;
    if (rank == size -1)
        n = N - n_ * (size-1);
    
    BPC(bop_ini(&d));
    BPC(bop_set_n(n, d));
    BPC(bop_set_type(BopFLOAT, d));
    BPC(bop_set_vars(2, "x y", d));
    BPC(bop_alloc(d));

    data = (float*) bop_get_data(d);

    for (i = 0; i < n; ++i) {
        i_ = rank * n_ + i;
        data[2*i + 0] = i_ * 0.5;
        data[2*i + 1] = N - i_ * 0.5;
    }

    BPC(bop_write_header(MPI_COMM_WORLD, "test", d));
    BPC(bop_write_values(MPI_COMM_WORLD, "test", d));
    
    BPC(bop_fin(d));

    MPI_Finalize();
    return 0;
}


/*

  # TEST: write.mpi.t0
  # mpirun -n 3 ./write_mpi
  # bop2txt test.bop > test.out.txt

*/
