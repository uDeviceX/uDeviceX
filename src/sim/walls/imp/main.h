static long get_max_parts(OptParams p) {
    int3 L = p.L;
    return p.numdensity *
        (L.x + 2 * XWM) *
        (L.y + 2 * YWM) *
        (L.z + 2 * ZWM);
}

void wall_gen(MPI_Comm cart, const Coords *coords, OptParams op, bool dump_sdf,
              /*io*/ int *n, Particle *pp, /**/ Wall *w) {
    long maxn, nold, nnew;
    if (!w) return;

    maxn = get_max_parts(op);
    nold = *n;
    UC(sdf_gen(coords, cart, dump_sdf, /**/ w->sdf));
    MC(m::Barrier(cart));
    UC(wall_gen_quants(cart, maxn, w->sdf, /**/ n, pp, &w->q));
    if (w->q.n) UC(wall_gen_ticket(&w->q, w->t));
    nnew = *n;
    msg_print("solvent particles survived: %d/%d", nnew, nold);
}

void wall_restart(MPI_Comm cart, const Coords *coords, OptParams op, bool dump_sdf,
                  const char *base, /**/ Wall *w) {
    long maxn;
    if (!w) return;

    maxn = get_max_parts(op);
    UC(sdf_gen(coords, cart, dump_sdf, /**/ w->sdf));
    MC(m::Barrier(cart));
    UC(wall_strt_quants(cart, base, maxn, &w->q));
    UC(wall_gen_ticket(&w->q, w->t));
}

void wall_dump_templ(const Wall *w, MPI_Comm cart, const char *base) {
    UC(wall_strt_dump_templ(cart, base, &w->q));
}

void wall_get_sdf_ptr(const Wall *w, const Sdf **s) {*s = w->sdf;}

double wall_compute_volume(const Wall *w, MPI_Comm comm, int3 L) {
    enum {NSAMPLES = 100000};
    return sdf_compute_volume(comm, L, w->sdf, NSAMPLES);
}

void wall_interact(const Coords *coords, const PairParams *par, Wall *w, PFarrays *aa) {
    long n, i, na;
    PaArray p;
    FoArray f;

    na = pfarrays_size(aa);

    for (i = 0; i < na; ++i) {
        UC(pfarrays_get(i, aa, &n, &p, &f));
        if (n) UC(wall_force(par, w->velstep, coords, w->sdf, &w->q, w->t, n, &p, /**/ &f));
    }
}

void wall_adhesion(const Coords *coords, const PairParams* params[], Wall *w, PFarrays *aa) {
    long n, i, na;
    PaArray p;
    FoArray f;
    const PairParams *par;

    na = pfarrays_size(aa);

    for (i = 0; i < na; ++i) {
        UC(pfarrays_get(i, aa, &n, &p, &f));
        par = params[i];
        if (n) UC(wall_force_adhesion(par, w->velstep, coords, w->sdf, &w->q, w->t, n, &p, /**/ &f));
    }
}

void wall_bounce(const Wall *w, const Coords *coords, float dt, PFarrays *aa) {
    long n, i, na;
    PaArray p;
    FoArray f;

    na = pfarrays_size(aa);

    for (i = 0; i < na; ++i) {
        UC(pfarrays_get(i, aa, &n, &p, &f));
        if (n) UC(sdf_bounce(dt, w->velstep, coords, w->sdf, n, /**/ (Particle*) p.pp));
    }    
}

void wall_repulse(const Wall *w, PFarrays *aa) {
    long n, i, na;
    PaArray p;
    FoArray f;

    na = pfarrays_size(aa);
    for (i = 0; i < na; ++i) {
        UC(pfarrays_get(i, aa, &n, &p, &f));
        if (n) UC(wall_repulse(w->sdf, n, &p, /**/ &f));
    }
}

void wall_update_vel(float t, Wall *w) {
    UC(wvel_get_step(t, w->vel, /**/ w->velstep));
}
