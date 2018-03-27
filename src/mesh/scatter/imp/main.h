void mesh_scatter_ini(MeshRead*, MeshScatter **pq) {
    MeshScatter *q;
    EMALLOC(1, &q);
    *pq = q;
}
