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

void send() {
  int dest = RECV;
  int buf[] = {42};
  MPI_Send(buf, send_cnt, MPI_INT, dest, TAG, COMM);
}


void recv() {
  int dest = SEND;
  int buf[123];
  MPI_Recv(buf, recv_cnt, MPI_INT, dest, TAG, COMM, status);
  printf("recv: %d\n", buf[0]);
}


int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(COMM, &rank);

  if (rank == SEND) send();
  else              recv();

  MPI_Finalize();
  return 0;
}
