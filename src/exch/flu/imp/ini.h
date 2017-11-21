
void ini(int maxd, Pack *p) {
    int i, nc, cap[NBAGS], ncs[NBAGS];
    size_t sz;

    frag_estimates(NFRAGS, maxd, /**/ cap);
    cap[BULK] = 0;
    
    for (i = 0; i < NFRAGS; ++i) {
        ncs[i] = nc = frag_ncell(i);
        sz = (nc + 1) * sizeof(int);
        d::Malloc((void**) &p->bcc.d[i], sz);
        d::Malloc((void**) &p->bss.d[i], sz);
    }
    ncs[BULK] = 0;
    
    ini(PINNED_DEV, NONE, sizeof(Particle), cap, /**/ &p->hpp, &p->dpp);
    ini(PINNED_DEV, NONE,      sizeof(int), cap, /**/ &p->hcc, &p->dcc);

    ini(PINNED_DEV, NONE, sizeof(int), ncs, /**/ &p->hfss, &p->dfss);
}
