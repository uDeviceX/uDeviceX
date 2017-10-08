void vel0(dev::Map m, int n, /**/ Particle *pp) {
    enum {X, Y, Z};

    float3 v;
    float  u[3];

    reini();
    KL(dev::sum, (k_cnf(n)), (m.color, n, pp, m.cc));

    avg_v(/**/ u);
    v = make_float3(u[X], u[Y], u[Z]);
    KL(dev::shift, (k_cnf(n)), (m.color, v, n, m.cc, /**/ pp));

    stat::setv(u);
}

void vel(const int *cc, int color, int n, /**/ Particle *pp) {
    dev::Map m;
    m.cc = cc; m.color = color;
    vel0(m, n, /**/ pp);
}
