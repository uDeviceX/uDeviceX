#include <stdio.h>
#include <mpi.h>

/* send.h */
void send();
void wait();

/* recv.h */
void recv(int*, int*);

#define SEND 0
#define RECV 1
#define COMM MPI_COMM_WORLD

int buf[123];
int count;
int rank;

void dump() {
    int i;
    for (i = 0; i < count; i++)
        fprintf(stderr, "%d ", buf[i]);
    fprintf(stderr, "\n");
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(COMM, &rank);

    if (rank == SEND) {
        send();
        wait();
    }
    else {
        recv(buf, &count);
        dump();
    }
    
    MPI_Finalize();
    return 0;
}
