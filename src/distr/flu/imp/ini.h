static int estimate_max_flux(int fid, int maxd) {
    int e, nfaces, d[] = frag_i2d3(fid);
    nfaces = abs(d[0]) + abs(d[1]) + abs(d[2]);
    e = maxd * frag_ncell(fid) * nfaces;
    return e;
}

static void get_capacity(int maxd, /**/ int capacity[NBAGS]) {
    for (int i = 0; i < NFRAGS; ++i)
        capacity[i] = estimate_max_flux(i, maxd);
    capacity[frag_bulk] = 0;
}

void ini(int maxdensity, Pack *p) {
    int capacity[NBAGS];
    get_capacity(maxdensity, /**/ capacity);

    UC(dmap_ini(NFRAGS, capacity, /**/ &p->map));
    
    UC(bags_ini(PINNED, NONE, sizeof(Particle), capacity, /**/ &p->hpp, &p->dpp));
    if (global_ids)    UC(bags_ini(PINNED, NONE, sizeof(int), capacity, /**/ &p->hii, &p->dii));
    if (multi_solvent) UC(bags_ini(PINNED, NONE, sizeof(int), capacity, /**/ &p->hcc, &p->dcc));
}

void ini(MPI_Comm comm, /**/ Comm *c) {
    UC(ini(comm, /**/ &c->pp));
    if (global_ids)    UC(ini(comm, /**/ &c->ii));
    if (multi_solvent) UC(ini(comm, /**/ &c->cc));
}

static int nhalocells() {
    return 8 +               /* corners */
        4 * (XS + YS + ZS) + /* edges   */
        2 * XS * YS +        /* faces   */
        2 * XS * ZS +
        2 * YS * ZS;
}

void ini(int maxdensity, Unpack *u) {
    int capacity[NBAGS];
    get_capacity(maxdensity, /**/ capacity);

    UC(bags_ini(HST_ONLY, NONE, sizeof(Particle), capacity, /**/ &u->hpp, NULL));
    if (global_ids)    UC(bags_ini(HST_ONLY, NONE, sizeof(int), capacity, /**/ &u->hii, NULL));
    if (multi_solvent) UC(bags_ini(HST_ONLY, NONE, sizeof(int), capacity, /**/ &u->hcc, NULL));

    int maxparts = (int) (nhalocells() * maxdensity) + 1;
    CC(d::Malloc((void**) &u->ppre, maxparts * sizeof(Particle)));
    if (global_ids)    CC(d::Malloc((void**) &u->iire, maxparts * sizeof(int)));
    if (multi_solvent) CC(d::Malloc((void**) &u->ccre, maxparts * sizeof(int)));
}
