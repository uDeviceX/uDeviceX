int Allreduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm) {
  return MPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
}

int Barrier(MPI_Comm comm) {
  return MPI_Barrier(comm);
}

int Cancel(MPI_Request *request) {
  return MPI_Cancel(request);
}

int Cart_rank(MPI_Comm comm, const int coords[], int *rank) {
  return MPI_Cart_rank(comm, coords, rank);
}

int Comm_dup(MPI_Comm comm, MPI_Comm *newcomm) {
  return MPI_Comm_dup(comm, newcomm);
}

int Comm_free(MPI_Comm *comm) {
  return MPI_Comm_free(comm);
}

int Exscan(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm) {
  return MPI_Exscan(sendbuf, recvbuf, count, datatype, op, comm);
}

int File_close(MPI_File *fh) {
  return MPI_File_close(fh);
}

int File_get_position(MPI_File fh, MPI_Offset *offset) {
  return MPI_File_get_position(fh, offset);
}

int File_open(MPI_Comm comm, const char *filename, int amode, MPI_Info info, MPI_File *fh) {
  return MPI_File_open(comm, filename, amode, info, fh);
}

int File_seek(MPI_File fh, MPI_Offset offset, int whence) {
  return MPI_File_seek(fh, offset, whence);
}

int File_set_size(MPI_File fh, MPI_Offset size) {
  return MPI_File_set_size(fh, size);
}

int File_write_at_all(MPI_File fh, MPI_Offset offset, const void *buf, int count, MPI_Datatype datatype, MPI_Status *status) {
  return MPI_File_write_at_all(fh, offset, buf, count, datatype, status);
}

int Finalize(void) {
  return MPI_Finalize();
}

int Irecv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Request *request) {
  return MPI_Irecv(buf, count, datatype, source, tag, comm, request);
}

int Isend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request) {
  return MPI_Isend(buf, count, datatype, dest, tag, comm, request);
}

int Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status *status) {
  return MPI_Recv(buf, count, datatype, source, tag, comm, status);
}

int Reduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm) {
  return MPI_Reduce(sendbuf, recvbuf, count, datatype, op, root, comm);
}

int Type_commit(MPI_Datatype *type) {
  return MPI_Type_commit(type);
}

int Type_contiguous(int count, MPI_Datatype oldtype, MPI_Datatype *newtype) {
  return MPI_Type_contiguous(count, oldtype, newtype);
}

int Waitall(int count, MPI_Request array_of_requests[], MPI_Status *array_of_statuses) {
  return MPI_Waitall(count, array_of_requests, array_of_statuses);
}
