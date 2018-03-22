void diag_part_ini(const char *path, /**/ DiagPart **pq) {
    FILE *f;
    DiagPart *q;
    EMALLOC(1, &q);
    snprintf(q->path, FILENAME_MAX, "%s/%s", DUMP_BASE, path);
    UC(efopen(q->path, "w", /**/ &f));
    UC(efclose(f));
    msg_print("DiagPart: %s", q->path);
    q->stamp = STAMP_GOOD;
    *pq = q;
}
void diag_part_fin(DiagPart *q) { EFREE(q); }

static float sq(float x) { return x*x; }
static int reduce(MPI_Comm comm, const void *sendbuf0, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op) {
    int root = 0;
    const void *sendbuf = (m::is_master(comm) ? MPI_IN_PLACE : sendbuf0);
    return m::Reduce(sendbuf, recvbuf, count, datatype, op, root, comm);
}
static int sum3(MPI_Comm comm, double *v) {
    return reduce(comm, v, m::is_master(comm) ? v : NULL, 3, MPI_DOUBLE, MPI_SUM);
}
static int sum_d(MPI_Comm comm, double *v) {
    return reduce(comm, v, v, 1, MPI_DOUBLE, MPI_SUM);
}
static int max_d(MPI_Comm comm, double *v) {
    return reduce(comm, v, v, 1, MPI_DOUBLE, MPI_MAX);
}
static int sum_i(MPI_Comm comm, int *v) {
    return reduce(comm, v, v, 1, MPI_INT, MPI_SUM);
}
void diag_part_apply(DiagPart *q, MPI_Comm comm, float time, int n, const Particle *pp) {
    if (time < 0) ERR("time = %g < 0", time);
    if (q->stamp != STAMP_GOOD) ERR("'q' was not initilized");
    enum {X, Y, Z};
    int i, c;
    double k, km, ke; /* particle, total, and maximum kinetic energies */
    double kbt = 0;
    FILE *f = NULL;
    double v[3] = {0};
    for (i = 0; i < n; ++i) for (c = 0; c < 3; ++c) v[c] += pp[i].v[c];
    sum3(comm, v);

    ke = km = 0;
    for (i = 0; i < n; ++i) {
        k = sq(pp[i].v[X]) + sq(pp[i].v[Y]) + sq(pp[i].v[Z]);
        ke += k;
        if (k > km) km = k;
    }

    sum_d(comm, &ke); max_d(comm, &km);
    sum_i(comm, &n);

    if (m::is_master(comm)) {
        if (n) kbt = 0.5 * ke / (n * 3. / 2);
        UC(efopen(q->path, "a", /**/ &f));
        msg_print("% .1e % .1e [% .1e % .1e % .1e] % .1e", time, kbt, v[X], v[Y], v[Z], km);
        fprintf(f, "%e\t%.10e\t%.10e\t%.10e\t%.10e\t%.10e\n", time, kbt, v[X], v[Y], v[Z], km);
        UC(efclose(f));
    }
}
