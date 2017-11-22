void fin(Pack *p) {
    for (int i = 0; i < NFRAGS; ++i) {
        d::Free(p->bcc.d[i]);
        d::Free(p->bss.d[i]);
        d::Free(p->bii.d[i]);
    }
    
    fin(PINNED_DEV, NONE, /**/ &p->hpp, &p->dpp);
    fin(PINNED_DEV, NONE, /**/ &p->hcc, &p->dcc);

    fin(PINNED_HST, NONE, /**/ &p->hfss, NULL);

    d::Free(p->counts_dev);
}

void fin(Comm *c) {
    fin(/**/ &c->pp);
    fin(/**/ &c->fss);
    if (multi_solvent)
        fin(/**/ &c->cc);
}
