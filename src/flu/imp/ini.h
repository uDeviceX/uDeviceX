static void ini_ii(int maxp, int **ii, int **ii0, int **ii_hst) {
    Dalloc(ii, maxp);
    Dalloc(ii0, maxp);
    EMALLOC(maxp, ii_hst);
}

void flu_ini(bool colors, bool ids, int3 L, int maxp, FluQuants *q) {
    q->n = 0;
    q->maxp = maxp;
    Dalloc(&q->pp, maxp);
    Dalloc(&q->pp0, maxp);
    UC(clist_ini(L.x, L.y, L.z, /**/ &q->cells));
    UC(clist_ini_map(maxp, 2, &q->cells, /**/ &q->mcells));

    EMALLOC(maxp, &q->pp_hst);

    if (ids)    ini_ii(maxp, &q->ii, &q->ii0, &q->ii_hst);
    if (colors) ini_ii(maxp, &q->cc, &q->cc0, &q->cc_hst);

    q->ids = ids;
    q->colors = colors;
}
