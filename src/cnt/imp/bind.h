void build_cells(int nw, const PaWrap *pw, /**/ Contact *c) {
    const PaWrap *w;
    int i, cc[MAX_OBJ_TYPES] = {0};

    clist::ini_counts(&c->cells);
    const bool project = true;

    for (i = 0; i < nw; ++i) {
        w = pw + i;
        cc[i] = w->n;
        clist::subindex(project, i, w->n, w->pp, /**/ &c->cells, &c->cmap);
    }
    clist::build_map(cc, /**/ &c->cells, &c->cmap);
}
