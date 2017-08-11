/* Usage: make run */
#include <stdio.h>
#include <mpi.h>

MPI_Status  status;
MPI_Request request;

#define send_cnt  1
#define recv_cnt 10

#define SEND 0 /* who sends and who receives? */
#define RECV 1

#define TAG 0
#define COMM MPI_COMM_WORLD

void send() {
  int dest = RECV;
  int buf[] = {42};
  MPI_Isend(buf, send_cnt, MPI_INT, dest, TAG, COMM, &request);
}

void recv() {
  int dest = SEND;
  int buf[123];
  MPI_Irecv(buf, recv_cnt, MPI_INT, dest, TAG, COMM, &request);
  MPI_Wait(&request, &status);
  printf("recv: %d\n", buf[0]);
}

int main(int argc, char *argv[]) {
  int rank;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(COMM, &rank);

  if (rank == SEND) send();
  else              recv();

  MPI_Finalize();
  return 0;
}
