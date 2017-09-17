
static void alloc_map(float maxdensity, /**/ Map *m) {
    CC(d::Malloc((void**) &m->counts,  NFRAGS      * sizeof(int)));
    CC(d::Malloc((void**) &m->starts, (NFRAGS + 1) * sizeof(int)));

    int e[NFRAGS], i;
    frag_estimates(NFRAGS, maxdensity, /**/ e);

    for (i = 0; i < NFRAGS; ++i)
        CC(d::Malloc((void**) &m->ids[i], e[i] * sizeof(int)));
}

void ini(float maxdensity, Pack *p) {
    alloc_map(maxdensity, /**/ &p->map);
    ini(PINNED, NONE, sizeof(Particle), maxdensity, /**/ &p->hpp, &p->dpp);
    if (global_ids)    ini(PINNED, NONE, sizeof(int), maxdensity, /**/ &p->hii, &p->dii);
    if (multi_solvent) ini(PINNED, NONE, sizeof(int), maxdensity, /**/ &p->hcc, &p->dcc);
}

void ini(MPI_Comm comm, /*io*/ basetags::TagGen *tg, /**/ Comm *c) {
    ini(comm, /*io*/ tg, /**/ &c->pp);
    if (global_ids)    ini(comm, /*io*/ tg, /**/ &c->ii);
    if (multi_solvent) ini(comm, /*io*/ tg, /**/ &c->cc);
}

static int nhalocells() {
    return 8 +               /* corners */
        4 * (XS + YS + ZS) + /* edges   */
        2 * XS * YS +        /* faces   */
        2 * XS * ZS +
        2 * YS * ZS;
}

void ini(float maxdensity, Unpack *u) {
    ini(HST_ONLY, NONE, sizeof(Particle), maxdensity, /**/ &u->hpp, NULL);
    if (global_ids)    ini(HST_ONLY, NONE, sizeof(int), maxdensity, /**/ &u->hii, NULL);
    if (multi_solvent) ini(HST_ONLY, NONE, sizeof(int), maxdensity, /**/ &u->hcc, NULL);

    int maxparts = (int) (nhalocells() * maxdensity) + 1;
    CC(d::Malloc((void**) &u->ppre, maxparts * sizeof(Particle)));
    if (global_ids)    CC(d::Malloc((void**) &u->iire, maxparts * sizeof(int)));
    if (multi_solvent) CC(d::Malloc((void**) &u->ccre, maxparts * sizeof(int)));
}
