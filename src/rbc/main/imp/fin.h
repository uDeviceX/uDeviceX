static void fin_common(Quants *q) {
    Dfree(q->pp);
    Dfree(q->av);

    Dfree(q->tri);
    Dfree(q->adj0);
    Dfree(q->adj1);

    free(q->tri_hst);
    free(q->pp_hst);
}

static void fin_ids(Quants *q) { free(q->ii);   }
static void fin_edg(Quants *q) { Dfree(q->shape.edg);  }
static void fin_rnd(Quants *q) { Dfree(q->shape.anti); }

void fin(Quants *q) {
    fin_common(q);
    if (rbc_ids) fin_ids(q);
    if (RBC_STRESS_FREE) fin_edg(q);
    if (RBC_RND)         fin_rnd(q);    
}
