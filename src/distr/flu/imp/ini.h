static void get_capacity(float maxdensity, /**/ int capacity[NBAGS]) {
    frag_estimates(NBAGS, maxdensity, /**/ capacity);
    capacity[frag_bulk] = 0;    
}

void ini(float maxdensity, Pack *p) {
    int capacity[NBAGS];
    get_capacity(maxdensity, /**/ capacity);

    ini_map(NFRAGS, capacity, /**/ &p->map);
    
    ini(PINNED, NONE, sizeof(Particle), capacity, /**/ &p->hpp, &p->dpp);
    if (global_ids)    ini(PINNED, NONE, sizeof(int), capacity, /**/ &p->hii, &p->dii);
    if (multi_solvent) ini(PINNED, NONE, sizeof(int), capacity, /**/ &p->hcc, &p->dcc);
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
    int capacity[NBAGS];
    get_capacity(maxdensity, /**/ capacity);

    ini(HST_ONLY, NONE, sizeof(Particle), capacity, /**/ &u->hpp, NULL);
    if (global_ids)    ini(HST_ONLY, NONE, sizeof(int), capacity, /**/ &u->hii, NULL);
    if (multi_solvent) ini(HST_ONLY, NONE, sizeof(int), capacity, /**/ &u->hcc, NULL);

    int maxparts = (int) (nhalocells() * maxdensity) + 1;
    CC(d::Malloc((void**) &u->ppre, maxparts * sizeof(Particle)));
    if (global_ids)    CC(d::Malloc((void**) &u->iire, maxparts * sizeof(int)));
    if (multi_solvent) CC(d::Malloc((void**) &u->ccre, maxparts * sizeof(int)));
}
