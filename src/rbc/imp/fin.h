static void fin_common(RbcQuants *q) {
    Dfree(q->pp);
    UC(area_volume_fin(q->area_volume));
    EFREE(q->pp_hst);
}

static void fin_ids(RbcQuants *q) { EFREE(q->ii);   }

void rbc_fin(RbcQuants *q) {
    if (rbc_ids) fin_ids(q);
    fin_common(q);
}
