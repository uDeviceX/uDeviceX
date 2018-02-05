static void reduce(MPI_Comm comm, const void *sendbuf0, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op) {
    int root = 0;
    const void *sendbuf = (m::is_master(comm) ? MPI_IN_PLACE : sendbuf0);
    MC(m::Reduce(sendbuf, recvbuf, count, datatype, op, root, comm));
}

static void max_float(MPI_Comm comm, /*io*/ float *v) {
    int count = 1;
    UC(reduce(comm, v, v, count, MPI_FLOAT, MPI_MAX));
}
