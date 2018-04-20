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
