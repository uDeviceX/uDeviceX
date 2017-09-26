static float3 vavg; /* average velocity */
static int       N;

static int reduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op) {
    return m::Allreduce(sendbuf, recvbuf, count, datatype, op, m::cart);
}

static int sum_f3(float *v) {
    const float vs[3] = {v[0], v[1], v[2]};
    return reduce(vs, v, 3, MPI_FLOAT, MPI_SUM);
}

static int sum_i(int *v) {
    const int vs = *v;
    return reduce(&vs, v, 1, MPI_INT, MPI_SUM);
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

static void setn(int n) { N = n; }
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
    setn(n);
    return make_float3(v[X], v[Y], v[Z]);
}

void vel(const int *cc, int n, int color, /**/ Particle *pp) {
    reini();
    KL(dev::sum_vel, (k_cnf(n)), (color, n, pp, cc));

    vavg = avg_v();

    KL(dev::shift_vel, (k_cnf(n)), (vavg, n, /**/ pp));
}

void stat(int *n, float *v) {
    enum {X, Y, Z};
    v[X] = vavg.x; v[Y] = vavg.y; v[Z] = vavg.z;
    *n = N;
}
