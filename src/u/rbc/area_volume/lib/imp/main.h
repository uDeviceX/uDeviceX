static void area_volume_hst(int nt, int nv, int nc, const Particle *pp, const int4 *tri, /**/ float *hst) {
    float *dev;
    Dalloc(&dev, 2*nc);
    area_volume::main(nt, nv, nc, pp, tri, /**/ dev);
    cD2H(hst, dev, 2*nc);
    Dfree(dev);
}

static void run0(rbc::Quants q, rbc::force::TicketT t) {
    float area, volume, av[2];
    area_volume_hst(q.nt, q.nv, q.nc, q.pp, q.tri, /**/ av);
    area = av[0]; volume = av[1];
    printf("%g %g\n", area, volume);
}

static void run1(const char *cell, const char *ic, rbc::Quants q) {
    Coords coords;
    coords_ini(m::cart, &coords);
    rbc::force::TicketT t;
    rbc::main::gen_quants(coords, m::cart, cell, ic, /**/ &q);
    rbc::force::gen_ticket(q, &t);
    run0(q, t);
    rbc::force::fin_ticket(&t);
    fin_coords(&coords);
}

void run(const char *cell, const char *ic) {
    rbc::Quants q;
    rbc::main::ini(&q);
    run1(cell, ic, q);
    rbc::main::fin(&q);
}
