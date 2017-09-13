void fin(/**/ Clist *c) {
    CC(d::Free(c->starts));
    CC(d::Free(c->counts));
}

void fin_work(Work *w) {
    scan::free_work(/**/ &w->scan);

    CC(d::Free(w->eelo));
    CC(d::Free(w->eere));
    CC(d::Free(w->ii));
}
