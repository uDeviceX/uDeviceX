void vel0(MPI_Comm comm, dev::Map m, int n, /**/ Particle *pp) {
    enum {X, Y, Z};

    float3 v;
    float  u[3];

    reini();
    KL(dev::sum, (k_cnf(n)), (m, n, pp));

    avg_v(comm, /**/ u);
    v = make_float3(u[X], u[Y], u[Z]);
    KL(dev::shift, (k_cnf(n)), (m, v, n, /**/ pp));

    stat::setv(u);
}
