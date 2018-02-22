void mesh_volume_ini(MeshRead *mesh, MeshVolume **pq) {
    int nv;
    MeshVolume *q;
    EMALLOC(1, &q);
    nv = mesh_get_nv(mesh);
    EMALLOC(3*nv, &q->rr);
    *pq = q;
}

void mesh_volume_fin(MeshVolume *q) {
    EFREE(q->rr);
    EFREE(q);
}
