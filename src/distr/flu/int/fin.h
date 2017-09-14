void fin(Pack *p) {
    free_map(/**/ &p->map);
    fin_pinned(/**/ &p->hpp, &p->dpp);

    if (global_ids)
        fin_pinned(/**/ &p->hii, &p->dii);

    if (multi_solvent)
        fin_pinned(/**/ &p->hcc, &p->dcc);

}

void fin(Comm *c);
void fin(Unpack *u);
