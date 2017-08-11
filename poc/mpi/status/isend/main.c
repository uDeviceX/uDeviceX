#include <stdio.h>
#include <mpi.h>

#define recv_cnt 10

#define SEND 0
#define RECV 1

#define TAG 0
#define COMM MPI_COMM_WORLD
#define TYPE MPI_INT

#define SZ(a) sizeof(a)/sizeof(a[0])

int buf[123];
int rank;
MPI_Status status;
int count;

MPI_Request req;

void send() {
    int dest = RECV;
    int buf[] = {1, 2, 3};
    MPI_Isend(buf, SZ(buf), TYPE, dest, TAG, COMM, &req);
}

void recv() {
    int source = SEND;
    MPI_Recv(buf, recv_cnt, TYPE, source, TAG, COMM, &status);
}

void cnt() {
    MPI_Get_count(&status, TYPE, &count);
    fprintf(stderr, "count: %d\n", count);
}

void dump() {
    int i;
    for (i = 0; i < count; i++)
        fprintf(stderr, "%d ", buf[i]);
    fprintf(stderr, "\n");
}

void wait() {
    MPI_Status s;
    MPI_Wait(&req, &s);
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(COMM, &rank);

    if (rank == SEND) {
        send();
        wait();
    }
    else {
        recv();
        cnt();
        dump();
    }
    
    MPI_Finalize();
    return 0;
}
