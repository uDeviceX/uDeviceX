void fin(/**/ Clist *c) {
    CC(d::Free(c->starts));
    CC(d::Free(c->counts));
}

void fin_map(Map *m) {
    scan::free_work(/**/ &m->scan);

    for (int i = 0; i < m->nA; ++i)
        CC(d::Free(m->ee[i]));
    CC(d::Free(m->ii));
}
