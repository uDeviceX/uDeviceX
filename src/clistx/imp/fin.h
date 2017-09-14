void fin(/**/ Clist *c) {
    CC(d::Free(c->starts));
    CC(d::Free(c->counts));
}

void fin_ticket(Ticket *t) {
    scan::free_work(/**/ &t->scan);

    CC(d::Free(t->eelo));
    CC(d::Free(t->eere));
    CC(d::Free(t->ii));
}
