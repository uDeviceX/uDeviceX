static void area_volume_hst(AreaVolume *area_volume, int nc, const Particle *pp, /**/ float *hst) {
    float *dev;
    Dalloc(&dev, 2*nc);
    UC(area_volume_compute(area_volume, nc, pp, /**/ dev));
    cD2H(hst, dev, 2*nc);
    Dfree(dev);
}

static void run0(RbcQuants q, RbcForce t) {
    float area, volume, av[2];
    UC(area_volume_hst(q.area_volume, q.nc, q.pp, /**/ av));
    area = av[0]; volume = av[1];
    printf("%g %g\n", area, volume);
}

static void run1(const char *cell, const char *ic, RbcQuants q) {
    Coords *coords;
    coords_ini(m::cart, &coords);
    RbcForce t;
    rbc_gen_quants(coords, m::cart, cell, ic, /**/ &q);
    rbc_force_gen(q, &t);
    UC(run0(q, t));
    rbc_force_fin(&t);
    coords_fin(coords);
}

void run(const char *cell, const char *ic) {
    RbcQuants q;
    UC(rbc_ini(&q));
    UC(run1(cell, ic, q));
    UC(rbc_fin(&q));
}
