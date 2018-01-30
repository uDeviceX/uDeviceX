static void ini_ii(int maxp, int **ii, int **ii0, int **ii_hst) {
    size_t sz;
    Dalloc(ii, maxp);
    Dalloc(ii0, maxp);
    sz = maxp * sizeof(int);
    UC(emalloc(sz, /**/ (void **) ii_hst));
}

void flu_ini(int3 L, int maxp, FluQuants *q) {
    size_t sz;
    q->n = 0;
    q->maxp = maxp;
    Dalloc(&q->pp, maxp);
    Dalloc(&q->pp0, maxp);
    clist_ini(L.x, L.y, L.z, /**/ &q->cells);
    clist_ini_map(maxp, 2, &q->cells, /**/ &q->mcells);

    sz = maxp * sizeof(Particle);
    UC(emalloc(sz, /**/ (void **) &q->pp_hst));

    if (global_ids)    ini_ii(maxp, &q->ii, &q->ii0, &q->ii_hst);
    if (multi_solvent) ini_ii(maxp, &q->cc, &q->cc0, &q->cc_hst);
}
