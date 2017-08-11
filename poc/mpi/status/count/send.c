#include <mpi.h>

#define COMM MPI_COMM_WORLD
#define SEND 0
#define RECV 1
#define TAG 0
#define TYPE MPI_INT
#define SZ(a) sizeof(a)/sizeof(a[0])

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
