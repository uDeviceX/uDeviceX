static void fin_common(RbcQuants *q) {
    Dfree(q->pp);
    Dfree(q->av);

    UC(area_volume_fin(q->area_volume));
    Dfree(q->adj0);
    Dfree(q->adj1);
    free(q->pp_hst);
}

static void fin_ids(RbcQuants *q) { free(q->ii);   }
static void fin_edg(RbcQuants *q) { Dfree(q->shape.edg);  }
static void fin_rnd(RbcQuants *q) { Dfree(q->shape.anti); }

void rbc_fin(RbcQuants *q) {
    fin_common(q);
    if (rbc_ids) fin_ids(q);
    if (RBC_STRESS_FREE) fin_edg(q);
    if (RBC_RND)         fin_rnd(q);    
}
