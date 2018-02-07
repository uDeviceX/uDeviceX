static void garea_volume(RbcQuants q, /**/ float *a, float *v) {
    int nc;
    AreaVolume *area_volume;
    const Particle *pp;
    float hst[2], *dev;
    nc = q.nc; pp = q.pp; area_volume = q.area_volume;
    UC(area_volume_compute(area_volume, nc, pp, /**/ &dev));
    cD2H(hst, dev, 2);
    *a = hst[0]; *v = hst[1];
}

static void dump(float dt, const Coords *coords, RbcQuants q, RbcForce t, MeshWrite *mesh_write) {
    int n;
    Particle *pp;
    float area, volume, area0, volume0;
    static int i = 0;
    n = q.nc * q.nv;
    UC(emalloc(n*sizeof(Particle), (void**)&pp));
    cD2H(pp, q.pp, q.n);
    UC(mesh_write_dump(mesh_write, m::cart, coords, q.nc, pp, i++));

    UC(rbc_force_stat(/**/ &area0, &volume0));
    UC(garea_volume(q, /**/ &area, &volume));
    msg_print("av: %g %g", area/area0, volume/volume0);
    diag(m::cart, dt*i, n, pp);
    UC(efree(pp));
}

static void body_force(long it, const Coords *coords, const BForce *bf, RbcQuants q, Force *f) {
    UC(bforce_apply(it, coords, rbc_mass, bf, q.n, q.pp, /**/ f));
}

static void run0(float dt, float te, const Coords *coords, int part_freq, const BForce *bforce,
                 MoveParams *moveparams, RbcQuants q, RbcForce t,
                 const RbcParams *par, RbcStretch *stretch, MeshWrite *mesh_write, Force *f) {
    long i;
    long nsteps = (long)(te / dt);
    msg_print("will take %ld steps", nsteps);
    for (i = 0; i < nsteps; i++) {
        Dzero(f, q.n);
        rbc_force_apply(dt, q, t, par, /**/ f);
        stretch::apply(q.nc, stretch, /**/ f);
        if (pushrbc) body_force(i, coords, bforce, q, /**/ f);
        scheme_move_apply(dt, moveparams, rbc_mass, q.n, f, q.pp);
        if (i % part_freq  == 0) dump(dt, coords, q, t, mesh_write);
#ifdef RBC_CLEAR_VEL
        scheme_move_clear_vel(q.n, /**/ q.pp);
#endif
    }
}

static void run1(float dt, float te, const Coords *coords, int part_freq, const BForce *bforce, MoveParams *moveparams, RbcQuants q, RbcForce t, const RbcParams *par, MeshWrite *mesh_write,  RbcStretch *stretch) {
    Force *f;
    Dalloc(&f, q.n);
    Dzero(f, q.n);
    UC(run0(dt, te, coords, part_freq, bforce, moveparams, q, t, par, stretch, mesh_write, f));
    Dfree(f);
}

static void run2(float dt, float te, const Coords *coords, int part_freq, const BForce *bforce, MoveParams *moveparams, OffRead *off, const char *ic, const RbcParams *par, MeshWrite *mesh_write, RbcQuants q) {
    RbcStretch *stretch;
    RbcForce t;
    rbc_gen_quants(coords, m::cart, off, ic, /**/ &q);
    UC(stretch::ini("rbc.stretch", q.nv, /**/ &stretch));
    rbc_force_gen(q, &t);
    run1(dt, te, coords, part_freq, bforce, moveparams, q, t, par, mesh_write, stretch);
    stretch::fin(stretch);
    rbc_force_fin(&t);
}

void run(float dt, float te, const Coords *coords, int part_freq, const BForce *bforce, MoveParams * moveparams, const char *cell, const char *ic, const RbcParams *par) {
    const char *directory = "r";
    RbcQuants q;
    OffRead *off;
    MeshWrite *mesh_write;
    
    UC(off_read(cell, /**/ &off));
    UC(mesh_write_ini_off(off, directory, /**/ &mesh_write));

    rbc_ini(off, &q);
    run2(dt, te, coords, part_freq, bforce, moveparams, off, ic, par, mesh_write, q);
    rbc_fin(&q);

    UC(mesh_write_fin(mesh_write));
    UC(off_fin(off));
}
