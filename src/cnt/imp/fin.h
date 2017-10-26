void fin(Contact *c) {
    clist::fin(/**/ &c->cells);
    clist::fin_map(/**/ &c->cmap);
    delete c->rgen;
}
