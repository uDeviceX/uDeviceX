void fin(Pack *p) {
    for (int i = 0; i < NFRAGS; ++i) {
        d::Free(p->bcc.d[i]);
        d::Free(p->bss.d[i]);
        d::Free(p->bii.d[i]);
    }
    
    fin(PINNED_DEV, NONE, /**/ &p->hpp, &p->dpp);
    fin(PINNED_DEV, NONE, /**/ &p->hcc, &p->dcc);

    fin(PINNED_DEV, NONE, /**/ &p->hfss, &p->dfss);
}

