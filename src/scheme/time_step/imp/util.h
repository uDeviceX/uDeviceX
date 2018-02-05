static void reduce(MPI_Comm comm, const void *sendbuf0, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op) {
    int root = 0;
    const void *sendbuf = (m::is_master(comm) ? MPI_IN_PLACE : sendbuf0);
    MC(m::Reduce(sendbuf, recvbuf, count, datatype, op, root, comm));
}

static void max_float(MPI_Comm comm, /*io*/ float *v) {
    int count = 1;
    UC(reduce(comm, v, v, count, MPI_FLOAT, MPI_MAX));
}

static float accel_max(MPI_Comm comm, TimeStepAccel *q) {
    int n, i;
    float accel, force, mass, max;
    Force *ff;
    max = 0;
    for (i = 0; i < q->k; i++) {
        mass = q->mm[i]; n = q->nn[i]; ff = q->fff[i];
        force = force_stat_max(n, ff);
        accel = force/mass;
        if (accel > max) max = accel;
    }
    UC(max_float(comm, &max));
    return max;
}
