#include <stdio.h>
#include <mpi.h>
#include "common.h"

/* send.h */
void send();
void wait();

/* recv.h */
void recv(int*, int*);

int a[123];
int n;
int rank;

void dump() {
    int i;
    for (i = 0; i < n; i++)
        fprintf(stderr, "%d ", a[i]);
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
        recv(a, &n);
        dump();
    }
    
    MPI_Finalize();
    return 0;
}
