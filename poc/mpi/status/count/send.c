#include <mpi.h>
#include "common.h"

static MPI_Request req;

void send() {
    int dest = RECV;
    int buf[] = {1, 2, 3};
    MPI_Isend(buf, SZ(buf), TYPE, dest, TAG, COMM, &req);
}

void wait() {
    MPI_Status s;
    MPI_Wait(&req, &s);
}
