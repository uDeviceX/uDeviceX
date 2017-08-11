#include <stdio.h>
#include <mpi.h>

#define COMM MPI_COMM_WORLD
#define SEND 0
#define RECV 1
#define TAG 0
#define TYPE MPI_INT
#define recv_cnt 10
#define SZ(a) sizeof(a)/sizeof(a[0])

MPI_Status status;

static void data(int *buf) {
    int source = SEND;
    MPI_Recv(buf, recv_cnt, TYPE, source, TAG, COMM, &status);
}

static void cnt(int *count) {
    MPI_Get_count(&status, TYPE, count);
    fprintf(stderr, "count: %d\n", *count);
}

void recv(int *buf, int *count) {
    data(buf);
    cnt(count);
}
