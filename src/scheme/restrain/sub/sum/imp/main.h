static void reduce(MPI_Comm comm, const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op) {
    MC(m::Allreduce(sendbuf, recvbuf, count, datatype, op, comm));
}

void f3(MPI_Comm comm, float *v) {
    const float vs[3] = {v[0], v[1], v[2]};
    reduce(comm, vs, v, 3, MPI_FLOAT, MPI_SUM);
}

void i(MPI_Comm comm, int *v) {
    const int vs = *v;
    reduce(comm, &vs, v, 1, MPI_INT, MPI_SUM);
}
