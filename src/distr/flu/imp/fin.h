static void free_map(/**/ Map *m) {
    CC(d::Free(m->counts));
    CC(d::Free(m->starts));    
    for (int i = 0; i < NFRAGS; ++i)
        CC(d::Free(m->ids[i]));
}

void fin(Pack *p) {
    free_map(/**/ &p->map);
    fin_pinned(/**/ &p->hpp, &p->dpp);

    if (global_ids)    fin_pinned(/**/ &p->hii, &p->dii);
    if (multi_solvent) fin_pinned(/**/ &p->hcc, &p->dcc);
}

void fin(Comm *c) {
    fin(&c->pp);

    if (global_ids)    fin(&c->ii);
    if (multi_solvent) fin(&c->cc);
}

void fin(Unpack *u) {
    fin(&u->hpp);
    if (global_ids)    fin(&u->hii);
    if (multi_solvent) fin(&u->hcc);

    CC(d::Free(u->ppre));
    if (global_ids)    CC(d::Free(u->iire));
    if (multi_solvent) CC(d::Free(u->ccre));
}
