void juelicher_ini(const MeshRead *cell, Juelicher **pq) {
    Juelicher *q;
    int md, nt, nv;
    const int4 *tt;
    EMALLOC(1, &q);
    nv = mesh_read_get_nv(cell);
    nt = mesh_read_get_nt(cell);
    md = mesh_read_get_md(cell);
    tt = mesh_read_get_tri(cell);
    
    UC(adj_ini(md, nt, nv, tt, /**/ &q->adj));
    UC(adj_view_ini(q->adj, /**/ &q->adj_v));

    *pq = q;
}

void juelicher_fin(Juelicher *q) {
    UC(adj_fin(q->adj));
    UC(adj_view_fin(q->adj_v));
    EFREE(q);
}
