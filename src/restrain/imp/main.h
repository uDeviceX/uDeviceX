static float3 vavg; /* average velocity */

static int reduce(const void *sendbuf0, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op) {
    int root = 0;
    const void *sendbuf = (m::rank == 0 ? MPI_IN_PLACE : sendbuf0);
    return m::Reduce(sendbuf, recvbuf, count, datatype, op, root, m::cart);
}

static int sum_f3(float *v) {
    return reduce(&v, m::rank == 0 ? &v : NULL, 3, MPI_FLOAT, MPI_SUM);
}

static int sum_i(int *v) {
    return reduce(v, v, 1, MPI_INT, MPI_SUM);
}

static void reini() {
    int zeroi = 0;
    float3 zerof3 = make_float3(0, 0, 0);
    CC(d::MemcpyToSymbol(&dev::sumv,   &zerof3, sizeof(float3)));
    CC(d::MemcpyToSymbol(&dev::indrop, &zeroi,  sizeof(int)));
}

/* device to host */
static void d2h(int *n, float v[3]) {
    enum {X, Y, Z};
    float3 u;
    CC(d::MemcpyFromSymbol(&u, &dev::sumv,    sizeof(float3)));
    CC(d::MemcpyFromSymbol(&n, &dev::indrop,  sizeof(int)));
    v[X] = u.x; v[Y] = u.y; v[Z] = u.z;
}

static float3 avg_v() {
    enum {X, Y, Z};
    int n;
    float v[3];
    d2h(&n, v);
    sum_i (&n);
    sum_f3( v);
    if (n) {
        v[X] /= n;
        v[Y] /= n;
        v[Z] /= n;
    }
    return make_float3(v[X], v[Y], v[Z]);
}

void vel(const int *cc, int n, int color, /**/ Particle *pp) {
    reini();
    KL(dev::sum_vel, (k_cnf(n)), (color, n, pp, cc));

    vavg = avg_v();

    KL(dev::shift_vel, (k_cnf(n)), (vavg, n, /**/ pp));
}

void vcm(float *v) {
    enum {X, Y, Z};
    v[X] = vavg.x; v[Y] = vavg.y; v[Z] = vavg.z;
}
