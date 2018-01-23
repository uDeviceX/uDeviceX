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

void dflu_pack_ini(int maxdensity, DFluPack **pack) {
    DFluPack *p;
    int capacity[NBAGS];

    UC(emalloc(sizeof(DFluPack), (void**) pack));
    p = *pack;
    
    get_capacity(maxdensity, /**/ capacity);

    UC(dmap_ini(NFRAGS, capacity, /**/ &p->map));
    
    UC(comm_bags_ini(PINNED, NONE, sizeof(Particle), capacity, /**/ &p->hpp, &p->dpp));
    if (global_ids)    UC(comm_bags_ini(PINNED, NONE, sizeof(int), capacity, /**/ &p->hii, &p->dii));
    if (multi_solvent) UC(comm_bags_ini(PINNED, NONE, sizeof(int), capacity, /**/ &p->hcc, &p->dcc));
}

void dflu_comm_ini(MPI_Comm cart, /**/ DFluComm **com) {
    DFluComm *c;
    UC(emalloc(sizeof(DFluComm), (void**) com));
    c = *com;
    UC(comm_ini(cart, /**/ &c->pp));
    if (global_ids)    UC(comm_ini(cart, /**/ &c->ii));
    if (multi_solvent) UC(comm_ini(cart, /**/ &c->cc));
}

static int nhalocells() {
    return 8 +               /* corners */
        4 * (XS + YS + ZS) + /* edges   */
        2 * XS * YS +        /* faces   */
        2 * XS * ZS +
        2 * YS * ZS;
}

void dflu_unpack_ini(int maxdensity, DFluUnpack **unpack) {
    int capacity[NBAGS];
    DFluUnpack *u;

    UC(emalloc(sizeof(DFluUnpack), (void**) unpack));
    u = *unpack;
    
    get_capacity(maxdensity, /**/ capacity);

    UC(comm_bags_ini(HST_ONLY, NONE, sizeof(Particle), capacity, /**/ &u->hpp, NULL));
    if (global_ids)    UC(comm_bags_ini(HST_ONLY, NONE, sizeof(int), capacity, /**/ &u->hii, NULL));
    if (multi_solvent) UC(comm_bags_ini(HST_ONLY, NONE, sizeof(int), capacity, /**/ &u->hcc, NULL));

    int maxparts = (int) (nhalocells() * maxdensity) + 1;
    CC(d::Malloc((void**) &u->ppre, maxparts * sizeof(Particle)));
    if (global_ids)    CC(d::Malloc((void**) &u->iire, maxparts * sizeof(int)));
    if (multi_solvent) CC(d::Malloc((void**) &u->ccre, maxparts * sizeof(int)));
}
