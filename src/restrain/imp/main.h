static float3 vavg; /* average velocity */

static void reini() {
    int zeroi = 0;
    float3 zerof3 = make_float3(0.f, 0.f, 0.f);
    CC(d::MemcpyToSymbol(&dev::sumv,   &zerof3, sizeof(float3)));
    CC(d::MemcpyToSymbol(&dev::indrop, &zeroi,  sizeof(int)));
}

static float3 avg_v() {
    int n; float v[3];
    CC(d::MemcpyFromSymbol( v, &dev::sumv, sizeof(float3)));
    CC(d::MemcpyFromSymbol(&n, &dev::indrop,  sizeof(int)));

    MC(m::Allreduce(&n, MPI_IN_PLACE, 1, MPI_INT  , MPI_SUM, m::cart));
    MC(m::Allreduce( v, MPI_IN_PLACE, 3, MPI_FLOAT, MPI_SUM, m::cart));

    enum {X, Y, Z};
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
