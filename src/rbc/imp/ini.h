static void ini_common(RbcQuants *q, const int4 *tt) {
    int nv, nt;
    nv = q->nv; nt = q->nt;
    Dalloc(&q->pp, MAX_CELL_NUM * nv);
    UC(emalloc(MAX_CELL_NUM * nv * sizeof(Particle), (void**) &q->pp_hst));
    UC(area_volume_ini(nv, nt, tt, MAX_CELL_NUM, /**/ &q->area_volume));
    Dalloc(&q->adj0, nv * RBCmd);
    Dalloc(&q->adj1, nv * RBCmd);
    Dalloc(&q->av, 2*MAX_CELL_NUM);
}

static void ini_ids(RbcQuants *q) { UC(emalloc(MAX_CELL_NUM * sizeof(int), (void**) &q->ii)); }
static void ini_edg(RbcQuants *q)  { Dalloc(&q->shape.edg,  q->nv * RBCmd); }
static void ini_anti(RbcQuants *q) { Dalloc(&q->shape.anti, q->nv * RBCmd); }

void rbc_ini(OffRead *cell, RbcQuants *q) {
    const int4 *tt;
    q->nv = off_get_nv(cell);
    q->nt = off_get_nt(cell);
    tt = off_get_tri(cell);

    q->n = q->nc = 0;
    UC(ini_common(q, tt));
    if (rbc_ids)         UC(ini_ids(q));
    if (RBC_STRESS_FREE) UC(ini_edg (q));
    if (RBC_RND)         UC(ini_anti(q));
}
