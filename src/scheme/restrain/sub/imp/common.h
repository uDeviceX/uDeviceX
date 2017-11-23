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

void stats(/**/ int *n, float *v) { stat::get(/**/ n, v); }
