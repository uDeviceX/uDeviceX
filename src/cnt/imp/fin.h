void fin() {
    delete g::indexes;
    scan::free_work(&g::ws);
    delete g::entries;
    delete g::rgen;

    Dfree(g::counts);
    Dfree(g::starts);
}

void fin(Contact *c) {
    clist::fin(/**/ &c->cells);
    clist::fin_map(/**/ &c->cmap);
    delete c->rgen;
}
