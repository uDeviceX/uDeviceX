void fin(/**/ Clist *c) {
    CC(d::Free(c->starts));
    CC(d::Free(c->counts));
}

void fin_map(Map *m) {
    scan::free_work(/**/ &m->scan);

    CC(d::Free(m->eelo));
    CC(d::Free(m->eere));
    CC(d::Free(m->ii));
}
