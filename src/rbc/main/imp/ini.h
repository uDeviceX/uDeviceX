void ini(Quants *q) {
    q->n = q->nc = 0;
    Dalloc(&q->pp, MAX_PART_NUM);
    q->pp_hst = (Particle*) malloc(MAX_PART_NUM * sizeof(Particle));

    q->nt = RBCnt;
    q->nv = RBCnv;

    Dalloc(&q->tri,  q->nt);
    Dalloc(&q->adj0, q->nv * RBCmd);
    Dalloc(&q->adj1, q->nv * RBCmd);

    q->tri_hst = (int4*) malloc(RBCnt * sizeof(int4));
    Dalloc(&q->av, 2*MAX_CELL_NUM);

    if (rbc_ids)
        q->ii = (int*) malloc(MAX_CELL_NUM * sizeof(int));
}
