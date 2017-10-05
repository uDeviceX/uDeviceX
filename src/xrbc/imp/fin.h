void free_quants(Quants *q) {
    Dfree(q->pp);
    Dfree(q->av);

    Dfree(q->tri);
    Dfree(q->adj0);
    Dfree(q->adj1);

    delete[] q->tri_hst;
    delete[] q->pp_hst;
}
