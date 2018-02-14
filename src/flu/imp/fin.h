static void fin_ii(int *ii, int *ii0, int *ii_hst) {
    CC(d::Free(ii));
    CC(d::Free(ii0));
    UC(efree(ii_hst));
}

void flu_fin(FluQuants *q) {
    CC(d::Free(q->pp));
    CC(d::Free(q->pp0));
    clist_fin(&q->cells);
    clist_fin_map(q->mcells);
    UC(efree(q->pp_hst));

    if (q->ids)    fin_ii(q->ii, q->ii0, q->ii_hst);
    if (q->colors) fin_ii(q->cc, q->cc0, q->cc_hst);
}
