void mesh_ini(MeshRead *mesh, /**/ Mesh **pq) {
    Mesh *q;
    int i, j, nv, nt, ne;
    const int4 *tt;
    int4 t;
    EMALLOC(1, &q);

    nv = mesh_read_get_nv(mesh);
    nt = mesh_read_get_nt(mesh);
    ne = mesh_read_get_ne(mesh);
    tt = mesh_read_get_tri(mesh);

    EMALLOC(3*nt, &q->tt);
    EMALLOC(2*ne, &q->ee);
    for (i = j = 0; i < nt; i++) {
        t = tt[i];
        q->tt[j++] = t.x; q->tt[j++] = t.y; q->tt[j++] = t.z;
    }

    q->nv = nv; q->nt = nt; q->ne = ne;
    *pq = q;
}

void mesh_fin(Mesh *q) {
    EFREE(q->tt);
    EFREE(q->ee);
    EFREE(q);
}
