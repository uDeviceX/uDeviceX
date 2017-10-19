#include <stdio.h>
#include <mpi.h>

int rank;
MPI_Status status[123];

#define send_cnt  1
#define recv_cnt 10

#define SEND 0
#define RECV 1

#define TAG 0
#define COMM MPI_COMM_WORLD

void send(MPI_Comm comm) {
  int dest = RECV;
  int buf[] = {42};
  MPI_Send(buf, send_cnt, MPI_INT, dest, TAG, comm);
}


void recv(MPI_Comm comm) {
  int dest = SEND;
  int buf[123];
  MPI_Recv(buf, recv_cnt, MPI_INT, dest, TAG, comm, /**/ status);
  printf("recv: %d\n", buf[0]);
}


int main(int argc, char *argv[]) {
  MPI_Comm comm;
    
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(COMM, &rank);
  MPI_Comm_dup(COMM, &comm);

  if (rank == SEND) send(comm);
  else              recv(comm);

  MPI_Finalize();
  return 0;
}
