static void reduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op) {
    MC(m::Allreduce(sendbuf, recvbuf, count, datatype, op, m::cart));
}

void sum_f3(float *v) {
    const float vs[3] = {v[0], v[1], v[2]};
    reduce(vs, v, 3, MPI_FLOAT, MPI_SUM);
}

void sum_i(int *v) {
    const int vs = *v;
    reduce(&vs, v, 1, MPI_INT, MPI_SUM);
}
