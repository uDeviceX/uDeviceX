static void ini_common(RbcQuants *q, const int4 *tt) {
    int nv, nt;
    nv = q->nv; nt = q->nt;
    Dalloc(&q->pp, MAX_CELL_NUM * nv);
    EMALLOC(MAX_CELL_NUM * nv, &q->pp_hst);
    UC(area_volume_ini(nv, nt, tt, MAX_CELL_NUM, /**/ &q->area_volume));
}

static void ini_ids(RbcQuants *q)  { EMALLOC(MAX_CELL_NUM, &q->ii); }
static void ini_anti(RbcQuants *q) { Dalloc(&q->shape.anti, q->nv * RBCmd); }

void rbc_ini(MeshRead *cell, RbcQuants *q) {
    int md;
    const int4 *tt;
    md    = RBCmd;
    q->nv = mesh_get_nv(cell);
    q->nt = mesh_get_nt(cell);
    tt = mesh_get_tri(cell);

    q->n = q->nc = 0;
    UC(ini_common(q, tt));
    if (rbc_ids)         UC(ini_ids(q));
    if (RBC_RND)         UC(ini_anti(q));
    UC(setup(md, q->nt, q->nv, tt, /**/ q));
}
