static void ini_common(Quants *q) {
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
}

static void ini_ids(Quants *q) { q->ii = (int*) malloc(MAX_CELL_NUM * sizeof(int)); }
static void ini_edg(Quants *q) { Dalloc(&q->shape.edg, q->nv * RBCmd); }

void ini(Quants *q) {
    ini_common(q);
    if (rbc_ids) ini_ids(q);
    if (RBC_STRESS_FREE) ini_edg(q);
}
