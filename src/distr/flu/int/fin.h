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

void fin(Unpack *u);
