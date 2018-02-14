static void garea_volume(RbcQuants *q, /**/ float *a, float *v) {
    int nc;
    AreaVolume *area_volume;
    const Particle *pp;
    float hst[2], *dev;
    nc = q->nc; pp = q->pp; area_volume = q->area_volume;
    UC(area_volume_compute(area_volume, nc, pp, /**/ &dev));
    cD2H(hst, dev, 2);
    *a = hst[0]; *v = hst[1];
}

static void dump(MPI_Comm cart, float dt, const Coords *coords, RbcQuants *q, RbcForce *t, MeshWrite *mesh_write) {
    int n;
    Particle *pp;
    float area, volume, area0, volume0;
    static int i = 0;
    n = q->nc * q->nv;
    UC(emalloc(n*sizeof(Particle), (void**)&pp));
    cD2H(pp, q->pp, q->n);
    UC(mesh_write_dump(mesh_write, cart, coords, q->nc, pp, i++));

    UC(rbc_force_stat(/**/ &area0, &volume0));
    UC(garea_volume(q, /**/ &area, &volume));
    msg_print("av: %g %g", area/area0, volume/volume0);
    diag(cart, dt*i, n, pp);
    UC(efree(pp));
}

static void body_force(long it, const Coords *coords, const BForce *bf, RbcQuants *q, Force *f) {
    UC(bforce_apply(it, coords, rbc_mass, bf, q->n, q->pp, /**/ f));
}

static void run0(MPI_Comm cart, float dt, float te, const Coords *coords, float part_freq, const BForce *bforce,
                 MoveParams *moveparams, RbcQuants *q, RbcForce *t,
                 const RbcParams *par, RbcStretch *stretch, MeshWrite *mesh_write, Force *f) {
    long i;
    Time *time;
    time_ini(0, &time);
    for (i = 0; time_current(time) < te; i++) {
        Dzero(f, q->n);
        rbc_force_apply(t, par, dt, q, /**/ f);
        stretch::apply(q->nc, stretch, /**/ f);
        if (pushrbc) body_force(i, coords, bforce, q, /**/ f);
        scheme_move_apply(dt, moveparams, rbc_mass, q->n, f, q->pp);
        if (time_cross(time, part_freq))
            dump(cart, dt, coords, q, t, mesh_write);
#ifdef RBC_CLEAR_VEL
        scheme_move_clear_vel(q->n, /**/ q->pp);
#endif
        time_next(time, dt);
    }
    time_fin(time);
}

static void run1(MPI_Comm cart, float dt, float te, const Coords *coords, int part_freq, const BForce *bforce, MoveParams *moveparams, RbcQuants *q, RbcForce *t, const RbcParams *par, MeshWrite *mesh_write,  RbcStretch *stretch) {
    Force *f;
    Dalloc(&f, q->n);
    Dzero(f, q->n);
    UC(run0(cart, dt, te, coords, part_freq, bforce, moveparams, q, t, par, stretch, mesh_write, f));
    Dfree(f);
}

static void run2(MPI_Comm cart, float dt, float te, int seed,
                 const Coords *coords, float part_freq,
                 const BForce *bforce, MoveParams *moveparams,
                 OffRead *off, const char *ic, const RbcParams *par, MeshWrite *mesh_write,
                 RbcQuants *q) {
    int nv;
    RbcStretch *stretch;
    RbcForce *t;
    nv = off_get_nv(off);
    rbc_gen_quants(coords, cart, off, ic, /**/ q);
    UC(stretch::ini("rbc.stretch", q->nv, /**/ &stretch));
    rbc_force_ini(nv, seed, &t);
    run1(cart, dt, te, coords, part_freq, bforce, moveparams, q, t, par, mesh_write, stretch);
    stretch::fin(stretch);
    rbc_force_fin(t);
}

void run(MPI_Comm cart, float dt, float te, int seed,
         const Coords *coords, float part_freq, const BForce *bforce, MoveParams * moveparams,
         const char *cell, const char *ic, const RbcParams *par) {
    const char *directory = "r";
    RbcQuants q;
    OffRead *off;
    MeshWrite *mesh_write;
    
    UC(off_read(cell, /**/ &off));
    UC(mesh_write_ini_off(off, directory, /**/ &mesh_write));

    rbc_ini(off, &q);
    run2(cart, dt, te, seed, coords, part_freq, bforce, moveparams, off, ic, par, mesh_write, &q);
    rbc_fin(&q);

    UC(mesh_write_fin(mesh_write));
    UC(off_fin(off));
}
