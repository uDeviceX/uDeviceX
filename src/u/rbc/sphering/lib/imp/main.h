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

static void dump(MPI_Comm cart, DiagPart *diagpart, float dt, const Coords *coords, RbcQuants *q, MeshWrite *mesh_write) {
    int n;
    Particle *pp;
    float area, volume, area0, volume0;
    static int i = 0;
    n = q->nc * q->nv;
    EMALLOC(n, &pp);
    cD2H(pp, q->pp, q->n);
    UC(mesh_write_particles(mesh_write, cart, coords, q->nc, pp, i++));

    UC(rbc_force_stat(/**/ &area0, &volume0));
    UC(garea_volume(q, /**/ &area, &volume));
    msg_print("av: %g %g", area/area0, volume/volume0);
    diag_part_apply(diagpart, cart, dt*i, n, pp);
    EFREE(pp);
}

static void body_force(float mass, const Coords *coords, const BForce *bf, RbcQuants *q, Force *f) {
    UC(bforce_apply(coords, mass, bf, q->n, q->pp, /**/ f));
}

static void run0(MPI_Comm cart, float dt, float mass, float te, const Coords *coords, float part_freq, const BForce *bforce,
                 RbcQuants *q, RbcForce *t,
                 const RbcParams *par, bool stretch, RbcStretch *fstretch, MeshWrite *mesh_write, Force *f) {
    long i;
    TimeLine *time;
    DiagPart *diagpart;
    diag_part_ini("diag.txt", /**/ &diagpart);
    time_line_ini(0, &time);
    for (i = 0; time_line_current(time) < te; i++) {
        Dzero(f, q->n);
        rbc_force_apply(t, par, dt, q, /**/ f);
        if (stretch) rbc_stretch_apply(q->nc, fstretch, /**/ f);
        body_force(mass, coords, bforce, q, /**/ f);
        scheme_move_apply(dt, mass, q->n, f, q->pp);
        if (time_line_cross(time, part_freq))
            dump(cart, diagpart, dt, coords, q, mesh_write);
#ifdef RBC_CLEAR_VEL
        scheme_move_clear_vel(q->n, /**/ q->pp);
#endif
        time_line_next(time, dt);
    }
    time_line_fin(time);
    diag_part_fin(diagpart);
}

static void run1(MPI_Comm cart, float dt, float mass, float te, const Coords *coords, int part_freq, const BForce *bforce, RbcQuants *q, RbcForce *t, const RbcParams *par, MeshWrite *mesh_write, bool stretch, RbcStretch *fstretch) {
    Force *f;
    Dalloc(&f, q->n);
    Dzero(f, q->n);
    UC(run0(cart, dt, mass, te, coords, part_freq, bforce, q, t, par, stretch, fstretch, mesh_write, f));
    Dfree(f);
}

static void run2(const Config *cfg, MPI_Comm cart, float dt, float mass, float te,
                 const Coords *coords, float part_freq,
                 const BForce *bforce, bool stretch,
                 MeshRead *off, const char *ic, const RbcParams *par, MeshWrite *mesh_write,
                 RbcQuants *q) {
    RbcStretch *rbcstretch;
    RbcForce *t;
    rbc_gen_quants(coords, cart, off, ic, /**/ q);

    if (stretch) rbc_stretch_ini("rbc.stretch", q->nv, &rbcstretch);
    UC(rbc_force_ini(off, &t));
    UC(rbc_force_set_conf(off, cfg, t));
    UC(run1(cart, dt, mass, te, coords, part_freq, bforce, q, t, par, mesh_write, stretch, rbcstretch));
    if (stretch) rbc_stretch_fin(rbcstretch);
    UC(rbc_force_fin(t));
}

void run(const Config *cfg, MPI_Comm cart, float dt, float mass, float te,
         const Coords *coords, float part_freq, const BForce *bforce,
         const char *cell, const char *ic, const RbcParams *par) {
    const char *directory = "r";
    RbcQuants q;
    MeshRead *off;
    MeshWrite *mesh_write;
    int stretch;
    UC(conf_lookup_bool(cfg, "rbc.stretch", &stretch));

    UC(mesh_read_ini_off(cell, /**/ &off));
    UC(mesh_write_ini_from_mesh(cart, CENTER, off, directory, /**/ &mesh_write));

    rbc_ini(false, off, &q);

    run2(cfg, cart, dt, mass, te, coords, part_freq, bforce, stretch, off, ic, par, mesh_write, &q);
    rbc_fin(&q);

    UC(mesh_write_fin(mesh_write));
    UC(mesh_read_fin(off));
}
