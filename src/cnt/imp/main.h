void ini(int rank, /**/ Contact *c) {
    clist::ini(XS, YS, ZS, /**/ &c->cells);
    clist::ini_map(MAX_PART_NUM, MAX_OBJ_TYPES, &c->cells, /**/ &c->cmap);
    UC(rnd_ini(7119 - rank, 187 + rank, 18278, 15674, /**/ &c->rgen));
}

void fin(Contact *c) {
    clist::fin(/**/ &c->cells);
    clist::fin_map(/**/ &c->cmap);
    UC(rnd_fin(c->rgen));
}

void build_cells(int nw, const PaWrap *pw, /**/ Contact *c) {
    const PaWrap *w;
    PartList lp;
    int i, cc[MAX_OBJ_TYPES] = {0};

    clist::ini_counts(&c->cells);
    const bool project = true;

    for (i = 0; i < nw; ++i) {
        w = pw + i;
        cc[i] = w->n;
        lp.pp = w->pp;
        lp.deathlist = NULL;
        clist::subindex(project, i, w->n, lp, /**/ &c->cells, &c->cmap);
    }
    clist::build_map(cc, /**/ &c->cells, &c->cmap);
}
