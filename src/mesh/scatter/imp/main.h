void mesh_scatter_ini(MeshRead *mesh, MeshScatter **pq) {
    MeshScatter *q;
    int i, j, nv, ne;
    int *ee;
    const int4 *dd;
    int4 d;
    EMALLOC(1, &q);
    
    nv = mesh_read_get_nv(mesh);
    ne = mesh_read_get_ne(mesh);
    dd = mesh_read_get_dih(mesh);
    EMALLOC(2*ne, &ee);
    for (j = i = 0; i < ne; i++) {
        d = dd[i];
        ee[j++] = d.y;
        ee[j++] = d.z;
    }

    q->nv = nv;
    q->ne = ne;
    q->ee = ee;
    *pq = q;
}

void mesh_scatter_fin(MeshScatter *q) {
    EFREE(q->ee);
    EFREE(q);
}
