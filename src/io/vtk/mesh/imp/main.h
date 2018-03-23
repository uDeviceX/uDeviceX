void mesh_ini(MeshRead *mesh, /**/ Mesh **pq) {
    Mesh *q;
    EMALLOC(1, &q);
    *pq = q;
}

void mesh_fin(Mesh *q) { EFREE(q); }
