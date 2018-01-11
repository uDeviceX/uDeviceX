static void ini_ii(int **ii, int **ii0, int **ii_hst) {
    size_t sz;
    Dalloc(ii, MAX_PART_NUM);
    Dalloc(ii0, MAX_PART_NUM);
    sz = MAX_PART_NUM * sizeof(int);
    UC(emalloc(sz, /**/ (void **) ii_hst));
}

void ini(FluQuants *q) {
    size_t sz;
    q->n = 0;
    Dalloc(&q->pp, MAX_PART_NUM);
    Dalloc(&q->pp0, MAX_PART_NUM);
    clist_ini(XS, YS, ZS, /**/ &q->cells);
    clist_ini_map(MAX_PART_NUM, 2, &q->cells, /**/ &q->mcells);
    sz = MAX_PART_NUM * sizeof(Particle);
    UC(emalloc(sz, /**/ (void **) &q->pp_hst));

    if (global_ids)    ini_ii(&q->ii, &q->ii0, &q->ii_hst);
    if (multi_solvent) ini_ii(&q->cc, &q->cc0, &q->cc_hst);
}
