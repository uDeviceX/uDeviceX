static void ini_common(RbcQuants *q, const int4 *tt) {
    int nv, nt;
    nv = q->nv; nt = q->nt;
    Dalloc(&q->pp, MAX_CELL_NUM * nv);
    EMALLOC(MAX_CELL_NUM * nv, &q->pp_hst);
    UC(area_volume_ini(nv, nt, tt, MAX_CELL_NUM, /**/ &q->area_volume));
}

static void ini_ids(RbcQuants *q)  { EMALLOC(MAX_CELL_NUM, &q->ii); }

void rbc_ini(bool ids, const MeshRead *cell, RbcQuants *q) {
    const int4 *tt;
    q->nv = mesh_get_nv(cell);
    q->nt = mesh_get_nt(cell);
    tt = mesh_get_tri(cell);

    q->n = q->nc = 0;
    q->ids = ids;
    UC(ini_common(q, tt));
    if (ids)         UC(ini_ids(q));
}
