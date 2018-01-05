void fin(Pack *p) {
    UC(dmap_fin(NFRAGS, /**/ &p->map));
    fin(PINNED, NONE, /**/ &p->hpp, &p->dpp);

    if (global_ids)    fin(PINNED, NONE, /**/ &p->hii, &p->dii);
    if (multi_solvent) fin(PINNED, NONE, /**/ &p->hcc, &p->dcc);
}

void fin(Comm *c) {
    fin(&c->pp);

    if (global_ids)    fin(&c->ii);
    if (multi_solvent) fin(&c->cc);
}

void fin(Unpack *u) {
    fin(HST_ONLY, NONE, &u->hpp, NULL);
    if (global_ids)    fin(HST_ONLY, NONE, &u->hii, NULL);
    if (multi_solvent) fin(HST_ONLY, NONE, &u->hcc, NULL);

    CC(d::Free(u->ppre));
    if (global_ids)    CC(d::Free(u->iire));
    if (multi_solvent) CC(d::Free(u->ccre));
}
