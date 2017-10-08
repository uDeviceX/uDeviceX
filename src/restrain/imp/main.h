void vel(const int *cc, int color, int n, /**/ Particle *pp) {
    enum {X, Y, Z};
    dev::Map m;
    m.cc = cc; m.color = color;

    float3 v;
    float  u[3];

    assert(color < 10);

    reini();
    KL(dev::sum, (k_cnf(n)), (color, n, pp, cc));

    avg_v(/**/ u);
    v = make_float3(u[X], u[Y], u[Z]);
    KL(dev::shift, (k_cnf(n)), (color, v, n, cc, /**/ pp));

    stat::setv(u);
}

