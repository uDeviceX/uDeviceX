static void ini_common(RbcQuants *q) {
    q->n = q->nc = 0;
    Dalloc(&q->pp, MAX_CELL_NUM * RBCnv);
    UC(emalloc(MAX_CELL_NUM * RBCnv * sizeof(Particle), (void**) &q->pp_hst));

    q->nt = RBCnt;
    q->nv = RBCnv;

    Dalloc(&q->tri,  q->nt);
    Dalloc(&q->adj0, q->nv * RBCmd);
    Dalloc(&q->adj1, q->nv * RBCmd);

    UC(emalloc(RBCnt * sizeof(int4), (void**) &q->tri_hst));
    Dalloc(&q->av, 2*MAX_CELL_NUM);
}

static void ini_ids(RbcQuants *q) { UC(emalloc(MAX_CELL_NUM * sizeof(int), (void**) &q->ii)); }
static void ini_edg(RbcQuants *q)  { Dalloc(&q->shape.edg,  q->nv * RBCmd); }
static void ini_anti(RbcQuants *q) { Dalloc(&q->shape.anti, q->nv * RBCmd); }

void ini(RbcQuants *q) {
    UC(ini_common(q));
    if (rbc_ids) UC(ini_ids(q));
    if (RBC_STRESS_FREE) UC(ini_edg (q));
    if (RBC_RND)         UC(ini_anti(q));
}
