#include <stdio.h>
#include <mpi.h>

#define recv_cnt 10

#define SEND 0
#define RECV 1

#define TAG 0
#define COMM MPI_COMM_WORLD
#define TYPE MPI_INT

#define NMSG 10000000
#define PFRQ 100

int buf[123];
int rank;
int count;

MPI_Request req;

void recv0() {
    MPI_Request request;
    MPI_Irecv(buf, recv_cnt, TYPE, SEND, TAG, COMM, &request);
}

void recv() {
    long i, dump;
    dump = NMSG/PFRQ;
    for (i = 0; i < NMSG; i++) {
        if (i % dump == 1) fputc('.', stderr);
        recv0();
    }
}

int send() { for(;;); }

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(COMM, &rank);

    if (rank == RECV)  recv();
    else               send();
   
    MPI_Finalize();
    return 0;
}
