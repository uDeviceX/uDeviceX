static void fin_ii(int *ii, int *ii0, int *ii_hst) {
    CC(d::Free(ii));
    CC(d::Free(ii0));
    EFREE(ii_hst);
}

void flu_fin(FluQuants *q) {
    CC(d::Free(q->pp));
    CC(d::Free(q->pp0));
    UC(clist_fin(&q->cells));
    UC(clist_fin_map(q->mcells));
    EFREE(q->pp_hst);

    if (q->ids)    UC(fin_ii(q->ii, q->ii0, q->ii_hst));
    if (q->colors) UC(fin_ii(q->cc, q->cc0, q->cc_hst));
}
