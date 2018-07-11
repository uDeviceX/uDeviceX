enum {
    CID_SUBINDEX,
    CN_SUBINDICES
};

void cnt_ini(int maxp, int rank, int3 L, int nobj, /**/ Contact **cnt) {
    EMALLOC(1, cnt);
    Contact *c = *cnt;

    c->L = L;
    c->nobj = nobj;

    EMALLOC(nobj, &c->cells);
    EMALLOC(nobj, &c->cmap);

    for (int i = 0; i < nobj; ++i) {
        clist_ini(L.x, L.y, L.z, /**/ &c->cells[i]);
        clist_ini_map(maxp, CN_SUBINDICES, c->cells[i], /**/ &c->cmap[i]);
    }
    UC(rnd_ini(7119 - rank, 187 + rank, 18278, 15674, /**/ &c->rgen));
}

void cnt_fin(Contact *c) {
    for (int i = 0; i < c->nobj; ++i) {
        clist_fin(/**/ c->cells[i]);
        clist_fin_map(/**/ c->cmap[i]);
    }
    UC(rnd_fin(c->rgen));
    EFREE(c->cells);
    EFREE(c->cmap);
    EFREE(c);
}

static void build_cells(const PaWrap *w, Clist *c, ClistMap *m) {
    static const bool project = true;
    PartList lp;
    const int cc[] = {w->n};

    lp.pp = w->pp;
    lp.deathlist = NULL;

    clist_ini_counts(c);
    clist_subindex(project, CID_SUBINDEX, w->n, lp, /**/ c, m);
    clist_build_map(cc, /**/ c, m);
}

void cnt_build_cells(int nw, const PaWrap *pw, /**/ Contact *c) {
    for (int i = 0; i < nw; ++i)
        build_cells(&pw[i], c->cells[i], c->cmap[i]);
}
