void alloc_quants(Quants *q) {
    q->n = q->nc = 0;
    Dalloc(&q->pp, MAX_PART_NUM);
    q->pp_hst = new Particle[MAX_PART_NUM];

    q->nt = RBCnt;
    q->nv = RBCnv;

    Dalloc(&q->tri,  q->nt);
    Dalloc(&q->adj0, q->nv * RBCmd);
    Dalloc(&q->adj1, q->nv * RBCmd);

    q->tri_hst = new int4[MAX_FACE_NUM];
    Dalloc(&q->av, 2*MAX_CELL_NUM);
}
