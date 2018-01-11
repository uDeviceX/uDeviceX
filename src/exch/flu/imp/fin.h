void fin(Pack *p) {
    for (int i = 0; i < NFRAGS; ++i) {
        CC(d::Free(p->bcc.d[i]));
        CC(d::Free(p->bss.d[i]));
        CC(d::Free(p->fss.d[i]));
        CC(d::Free(p->bii.d[i]));
    }
    
    UC(bags_fin(PINNED_DEV, NONE, /**/ &p->hpp, &p->dpp));
    if (multi_solvent)
        UC(bags_fin(PINNED_DEV, NONE, /**/ &p->hcc, &p->dcc));

    UC(bags_fin(PINNED_HST, NONE, /**/ &p->hfss, NULL));

    CC(d::Free(p->counts_dev));
}

void fin(Comm *c) {
    fin(/**/ &c->pp);
    fin(/**/ &c->fss);
    if (multi_solvent)
        fin(/**/ &c->cc);
}

void fin(Unpack *u) {
    UC(bags_fin(PINNED_DEV, NONE, /**/ &u->hpp, &u->dpp));
    if (multi_solvent)
        UC(bags_fin(PINNED_DEV, NONE, /**/ &u->hcc, &u->dcc));

    UC(bags_fin(PINNED_HST, NONE, /**/ &u->hfss, &u->dfss));
}

