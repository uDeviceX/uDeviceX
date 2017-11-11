void fin(Quants *q) {
    Dfree(q->pp);
    Dfree(q->av);

    Dfree(q->tri);
    Dfree(q->adj0);
    Dfree(q->adj1);

    free(q->tri_hst);
    free(q->pp_hst);

    if (rbc_ids)
        free(q->ii);
}
