#include <stdio.h>
#include <mpi.h>
#include "common.h"

#define recv_cnt 10
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
