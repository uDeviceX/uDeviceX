void cnt_ini(int maxp, int rank, int3 L, /**/ Contact **cnt) {
    EMALLOC(1, cnt);
    Contact *c = *cnt;

    c->L = L;
    clist_ini(L.x, L.y, L.z, /**/ &c->cells);
    clist_ini_map(maxp, MAX_OBJ_TYPES, &c->cells, /**/ &c->cmap);
    UC(rnd_ini(7119 - rank, 187 + rank, 18278, 15674, /**/ &c->rgen));
}

void cnt_fin(Contact *c) {
    clist_fin(/**/ &c->cells);
    clist_fin_map(/**/ c->cmap);
    UC(rnd_fin(c->rgen));
    EFREE(c);
}

void cnt_build_cells(int nw, const PaWrap *pw, /**/ Contact *c) {
    const PaWrap *w;
    PartList lp;
    int i, cc[MAX_OBJ_TYPES] = {0};

    clist_ini_counts(&c->cells);
    const bool project = true;

    for (i = 0; i < nw; ++i) {
        w = pw + i;
        cc[i] = w->n;
        lp.pp = w->pp;
        lp.deathlist = NULL;
        clist_subindex(project, i, w->n, lp, /**/ &c->cells, c->cmap);
    }
    clist_build_map(cc, /**/ &c->cells, c->cmap);
}
