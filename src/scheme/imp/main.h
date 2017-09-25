void move(float mass, Particle *pp, Force *ff, int n) {
    KL(dev::update, (k_cnf(n)), (mass, pp, ff, n));
}

void clear_vel(Particle *pp, int n) {
    KL(dev::clear_vel, (k_cnf(n)), (pp, n));
}

void force(float mass, Particle* pp, Force* ff, int n, float driving_force0) {
    KL(dev::force, (k_cnf(n)), (mass, pp, ff, n, driving_force0));
}


static void reini() {
    int zeroi = 0;
    float3 zerof3 = make_float3(0.f, 0.f, 0.f);
    CC(d::MemcpyToSymbol(&dev::rd::sumv,   &zerof3, sizeof(float3)));
    CC(d::MemcpyToSymbol(&dev::rd::indrop, &zeroi,  sizeof(int)));
}

static float3 avg_v() {
    int n; float v[3];
    CC(d::MemcpyFromSymbol( v, &dev::rd::sumv, sizeof(float3)));
    CC(d::MemcpyFromSymbol(&n, &dev::rd::indrop,  sizeof(int)));

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

void restrain_drop_vel(int color, int n, Particle *pp, const int *cc) {
    reini();
    KL(dev::rd::sum_vel, (k_cnf(n)), (color, n, pp, cc));

    float3 vavg = avg_v();

    KL(dev::rd::shift_vel, (k_cnf(n)), (vavg, n, /**/ pp));
}
