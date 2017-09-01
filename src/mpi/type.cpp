#include <mpi.h>

#include "inc/conf.h"
#include "utils/mc.h"

#include "inc/type.h"
#include "mpi/type.h"

namespace datatype {
MPI_Datatype particle, partforce, solid;

void ini() {
    MC(MPI_Type_contiguous(sizeof(Particle) / sizeof(float), MPI_FLOAT, &particle));
    MC(MPI_Type_contiguous((sizeof(Particle) + sizeof(Force)) / sizeof(float), MPI_FLOAT, &partforce));
    MC(MPI_Type_contiguous(sizeof(Solid)    / sizeof(float), MPI_FLOAT, &solid));

    MC(MPI_Type_commit(&particle));
    MC(MPI_Type_commit(&partforce));
    MC(MPI_Type_commit(&solid));
}

void fin() {
    MC(MPI_Type_free(&particle));
    MC(MPI_Type_free(&partforce));
    MC(MPI_Type_free(&solid));
}
}
