namespace g {
static float3 v;
static int    n;
}

static void reini() {
    int zeroi = 0;
    float3 zerof3 = make_float3(0, 0, 0);
    CC(d::MemcpyToSymbol(&dev::g::v, &zerof3, sizeof(float3)));
    CC(d::MemcpyToSymbol(&dev::g::n, &zeroi,  sizeof(int)));
}

/* device to host */
static void d2h(int *n, float v[3]) {
    enum {X, Y, Z};
    float3 u;
    CC(d::MemcpyFromSymbol(&u, &dev::g::v, sizeof(float3)));
    CC(d::MemcpyFromSymbol( n, &dev::g::n, sizeof(int)));
    v[X] = u.x; v[Y] = u.y; v[Z] = u.z;
}

static void setn(int n)    { g::n = n; }
static void setv(float3 v) { g::v = v; }
static float3 avg_v() {
    enum {X, Y, Z};
    int n;
    float v[3];
    d2h(&n, v);
    sum::i (&n);
    sum::f3( v);
    if (n) {
        v[X] /= n;
        v[Y] /= n;
        v[Z] /= n;
    }
    setn(n);
    return make_float3(v[X], v[Y], v[Z]);
}

void vel(const int *cc, int n, int color, /**/ Particle *pp) {
    float3 v;
    reini();
    KL(dev::sum_vel, (k_cnf(n)), (color, n, pp, cc));

    v = avg_v();
    KL(dev::shift_vel, (k_cnf(n)), (color, v, n, cc, /**/ pp));
    setv(v);
}

void stat(int *n, float *v) { /* report statistics */
    enum {X, Y, Z};
    v[X] = g::v.x; v[Y] = g::v.y; v[Z] = g::v.z;
    *n = g::n;
}
