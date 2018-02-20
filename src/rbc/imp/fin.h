static void fin_common(RbcQuants *q) {
    Dfree(q->pp);
    UC(area_volume_fin(q->area_volume));
    EFREE(q->pp_hst);
}

static void fin_ids(RbcQuants *q) { EFREE(q->ii);   }
static void fin_rnd(RbcQuants *q) { Dfree(q->shape.anti); }

void rbc_fin(RbcQuants *q) {
    fin_common(q);
    if (rbc_ids) fin_ids(q);
    if (RBC_RND)         fin_rnd(q);    
}
