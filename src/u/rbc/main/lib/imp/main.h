static void garea_volume(RbcQuants q, /**/ float *a, float *v) {
    int nc;
    AreaVolume *area_volume;
    const Particle *pp;
    float hst[2], *dev;

    Dalloc(&dev, 2);

    nc = q.nc; pp = q.pp; area_volume = q.area_volume;
    UC(area_volume_compute(area_volume, nc, pp, /**/ dev));
    cD2H(hst, dev, 2);
    Dfree(dev);

    *a = hst[0]; *v = hst[1];
}

static void dump(const Coords *coords, RbcQuants q, RbcForce t) {
    int n;
    Particle *pp;
    float area, volume, area0, volume0;
    static int i = 0;
    n = q.nc * q.nv;
    UC(emalloc(n*sizeof(Particle), (void**)&pp));
    cD2H(pp, q.pp, q.n);
    mesh_write_rbc(m::cart, coords, pp, q.tri_hst, q.nc, q.nv, q.nt, i++);
    UC(rbc_force_stat(/**/ &area0, &volume0));
    UC(garea_volume(q, /**/ &area, &volume));
    msg_print("av: %g %g", area/area0, volume/volume0);
    diagnostics(m::cart, n, pp, i);
    free(pp);
}

static int body_force(const Coords *coords, const BForce *bf, RbcQuants q, Force *f) {
    UC(bforce_apply(0, coords, rbc_mass, bf, q.n, q.pp, /**/ f));
    return 0;
}

static void run0(const Coords *coords, int part_freq, const BForce *bforce, MoveParams *moveparams, RbcQuants q, RbcForce t, const RbcParams *par, RbcStretch* stretch, Force *f) {
    long i;
    long nsteps = (long)(tend / dt);
    msg_print("will take %ld steps", nsteps);
    for (i = 0; i < nsteps; i++) {
        Dzero(f, q.n);
        rbc_force_apply(q, t, par, /**/ f);
        stretch::apply(q.nc, stretch, /**/ f);
        if (pushrbc) body_force(coords, bforce, q, /**/ f);
        scheme::move::main(moveparams, rbc_mass, q.n, f, q.pp);
        if (i % part_freq  == 0) dump(coords, q, t);
#ifdef RBC_CLEAR_VEL
        scheme::move::clear_vel(q.n, /**/ q.pp);
#endif
    }
}

static void run1(const Coords *coords, int part_freq, const BForce *bforce, MoveParams *moveparams, RbcQuants q, RbcForce t, const RbcParams *par,
                 RbcStretch *stretch) {
    Force *f;
    Dalloc(&f, q.n);
    Dzero(f, q.n);
    run0(coords, part_freq, bforce, moveparams, q, t, par, stretch, f);
    Dfree(f);
}

static void run2(const Coords *coords, int part_freq, const BForce *bforce, MoveParams *moveparams, const char *cell, const char *ic, const RbcParams *par, RbcQuants q) {
    RbcStretch *stretch;
    RbcForce t;
    rbc_gen_quants(coords, m::cart, cell, ic, /**/ &q);
    UC(stretch::ini("rbc.stretch", q.nv, /**/ &stretch));
    rbc_force_gen(q, &t);
    run1(coords, part_freq, bforce, moveparams, q, t, par, stretch);
    stretch::fin(stretch);
    rbc_force_fin(&t);
}

void run(const Coords *coords, int part_freq, const BForce *bforce, MoveParams * moveparams, const char *cell, const char *ic, const RbcParams *par) {
    RbcQuants q;
    rbc_ini(&q);
    run2(coords, part_freq, bforce, moveparams, cell, ic, par, q);
    rbc_fin(&q);
}
