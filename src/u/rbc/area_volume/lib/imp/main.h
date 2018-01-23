static void area_volume_hst(int nt, int nv, int nc, const Particle *pp, const int4 *tri, /**/ float *hst) {
    float *dev;
    Dalloc(&dev, 2*nc);
    area_volume::main(nt, nv, nc, pp, tri, /**/ dev);
    cD2H(hst, dev, 2*nc);
    Dfree(dev);
}

static void run0(RbcQuants q, RbcForce t) {
    float area, volume, av[2];
    area_volume_hst(q.nt, q.nv, q.nc, q.pp, q.tri, /**/ av);
    area = av[0]; volume = av[1];
    printf("%g %g\n", area, volume);
}

static void run1(const char *cell, const char *ic, RbcQuants q) {
    Coords *coords;
    coords_ini(m::cart, &coords);
    RbcForce t;
    rbc_gen_quants(coords, m::cart, cell, ic, /**/ &q);
    rbc_force_gen(q, &t);
    run0(q, t);
    rbc_force_fin(&t);
    coords_fin(coords);
}

void run(const char *cell, const char *ic) {
    RbcQuants q;
    rbc_ini(&q);
    run1(cell, ic, q);
    rbc_fin(&q);
}
