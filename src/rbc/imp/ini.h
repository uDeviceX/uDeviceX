static void ini_common(long maxnc, const int4 *tt, RbcQuants *q) {
    int nv, nt;
    nv = q->nv; nt = q->nt;
    Dalloc(&q->pp, maxnc * nv);
    EMALLOC(maxnc * nv, &q->pp_hst);
    UC(area_volume_ini(nv, nt, tt, maxnc, /**/ &q->area_volume));
}

static void ini_ids(long maxnc, RbcQuants *q)  { EMALLOC(maxnc, &q->ii); }

void rbc_ini(bool ids, const MeshRead *cell, RbcQuants *q) {
    const int4 *tt;
    long maxnc = MAX_CELL_NUM;
    q->nv = mesh_read_get_nv(cell);
    q->nt = mesh_read_get_nt(cell);
    q->md = mesh_read_get_md(cell);
    tt = mesh_read_get_tri(cell);

    q->n = q->nc = 0;
    q->ids = ids;
    UC(ini_common(maxnc, tt, q));
    if (ids) UC(ini_ids(maxnc, q));
}
