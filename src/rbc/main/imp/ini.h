static void ini_common(Quants *q) {
    q->n = q->nc = 0;
    Dalloc(&q->pp, MAX_PART_NUM);
    emalloc(MAX_PART_NUM * sizeof(Particle), (void**) &q->pp_hst);

    q->nt = RBCnt;
    q->nv = RBCnv;

    Dalloc(&q->tri,  q->nt);
    Dalloc(&q->adj0, q->nv * RBCmd);
    Dalloc(&q->adj1, q->nv * RBCmd);

    emalloc(RBCnt * sizeof(int4), (void**) &q->tri_hst);
    Dalloc(&q->av, 2*MAX_CELL_NUM);
}

static void ini_ids(Quants *q) { emalloc(MAX_CELL_NUM * sizeof(int), (void**) &q->ii); }
static void ini_edg(Quants *q)  { Dalloc(&q->shape.edg,  q->nv * RBCmd); }
static void ini_anti(Quants *q) { Dalloc(&q->shape.anti, q->nv * RBCmd); }

void ini(Quants *q) {
    ini_common(q);
    if (rbc_ids) ini_ids(q);
    if (RBC_STRESS_FREE) ini_edg (q);
    if (RBC_RND)         ini_anti(q);
}
