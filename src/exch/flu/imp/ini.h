void ini(int maxd, Pack *p) {
    int i, nc, cap[NBAGS], ncs[NBAGS];
    size_t sz;

    frag_estimates(NFRAGS, maxd, /**/ cap);
    cap[BULK] = 0;
    memcpy(p->cap.d, cap, NFRAGS * sizeof(int));
    
    for (i = 0; i < NFRAGS; ++i) {
        ncs[i] = nc = frag_ncell(i) + 1;
        sz = nc * sizeof(int);
        CC(d::Malloc((void**) &p->bcc.d[i], sz));
        CC(d::Malloc((void**) &p->bss.d[i], sz));
        CC(d::Malloc((void**) &p->fss.d[i], sz));
        
        sz = cap[i] * sizeof(int);
        CC(d::Malloc((void**) &p->bii.d[i], sz));
    }
    ncs[BULK] = 0;
    
    UC(bags_ini(PINNED_DEV, NONE, sizeof(Particle), cap, /**/ &p->hpp, &p->dpp));
    if (multi_solvent)
        UC(bags_ini(PINNED_DEV, NONE,  sizeof(int), cap, /**/ &p->hcc, &p->dcc));

    UC(bags_ini(PINNED_HST, NONE, sizeof(int), ncs, /**/ &p->hfss, NULL));

    memcpy(p->hfss.counts, ncs, sizeof(ncs));
    
    sz = 26 * sizeof(int);
    CC(d::Malloc((void**) &p->counts_dev, sz));
}

void ini(MPI_Comm comm, /**/ Comm *c) {
    UC(ini(comm, /**/ &c->pp));
    UC(ini(comm, /**/ &c->fss));
    if (multi_solvent)
        UC(ini(comm, /**/ &c->cc));
}

void ini(int maxd, Unpack *u) {
    int i, cap[NBAGS], ncs[NBAGS];

    frag_estimates(NFRAGS, maxd, /**/ cap);
    cap[BULK] = 0;
    
    for (i = 0; i < NFRAGS; ++i)
        ncs[i] = frag_ncell(i) + 1;
    ncs[BULK] = 0;
    
    UC(bags_ini(PINNED_DEV, NONE, sizeof(Particle), cap, /**/ &u->hpp, &u->dpp));
    if (multi_solvent)
        UC(bags_ini(PINNED_DEV, NONE,  sizeof(int), cap, /**/ &u->hcc, &u->dcc));

    UC(bags_ini(PINNED_DEV, NONE, sizeof(int), ncs, /**/ &u->hfss, &u->dfss));
}

