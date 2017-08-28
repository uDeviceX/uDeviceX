namespace fsi {
static char buf[BUFSIZ];
#define F(s) __FILE__, __LINE__, s

static void bulk0(ParticlesWrap *w) {
    int n0, n1;
    float rnd;
    const Particle* pp  = w->p;

    rnd = rgen->get_float();
    n0 = w->n;
    n1 = wo->n;

    dbg::check_vv(pp, n0, F("B"));
    dbg::check_pos_pu(pp, n0, F("B"));
    KL(dev::bulk, (k_cnf(3*n0)), ((float2*)pp, n0, n1, rnd, (float*)w->f, (float*)wo->f));
}

void bulk(std::vector<ParticlesWrap> wr) {
    int i, n;
    ParticlesWrap *w; /* wrap */
    n = wr.size();
    w = wr.data();

    if (n == 0) return;
    setup(wo->p, wo->n, wo->cellsstart);
    for (i = 0; i < n; i++) bulk0(w++);
}
}
