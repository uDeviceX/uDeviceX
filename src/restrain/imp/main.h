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

static void avg_v(/**/ float *v) {
    enum {X, Y, Z};
    int n;
    d2h(&n, v);
    sum::i (&n);
    sum::f3( v);
    if (n) {
        v[X] /= n;
        v[Y] /= n;
        v[Z] /= n;
    }
    stat::setn(n);
}

void vel(const int *cc, int n, int color, /**/ Particle *pp) {
    enum {X, Y, Z};

    float3 v;
    float  u[3];

    reini();
    KL(dev::sum_vel, (k_cnf(n)), (color, n, pp, cc));

    avg_v(/**/ u);
    v = make_float3(u[X], u[Y], u[Z]);
    KL(dev::shift_vel, (k_cnf(n)), (color, v, n, cc, /**/ pp));

    stat::setv(u);
}

void stat(/**/ int *n, float *v) { stat::get(/**/ n, v); }
