void clist_fin(/**/ Clist *c) {
    Dfree(c->starts);
    Dfree(c->counts);
}

void clist_fin_map(ClistMap *m) {
    UC(scan_fin(/**/ m->scan));

    for (int i = 0; i < m->nA; ++i) Dfree(m->ee[i]);
    Dfree(m->ii);
    EFREE(m);
}
