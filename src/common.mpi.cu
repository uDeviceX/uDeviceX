#include <mpi.h>
#include "common.mpi.h"

namespace datatype {
MPI_Datatype particle, solid;

void ini() {
    MC(MPI_Type_contiguous(6, MPI_FLOAT, &particle));
    MC(MPI_Type_contiguous(32, MPI_FLOAT, &solid));

    MC(MPI_Type_commit(&particle));
    MC(MPI_Type_commit(&solid));
}

void fin() {
    MC(MPI_Type_free(&particle));
    MC(MPI_Type_free(&solid));
}
}
