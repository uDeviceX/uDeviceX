void fin(/**/ Clist *c) {
    CC(d::Free(c->starts));
    CC(d::Free(c->counts));
}

void fin_work(Work *w) {
    
}
