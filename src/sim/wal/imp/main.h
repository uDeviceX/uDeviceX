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
    UC(wall_gen_ticket(&w->q, w->t));
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
