void diag_mesh_ini(const char *path, MeshRead *mesh, DiagMesh **pq) {
    FILE *f;
    DiagMesh *q;
    EMALLOC(1, &q);
    *pq = q;
    q->nv = mesh_get_nv(mesh);
    q->nt = mesh_get_nt(mesh);
    EMALLOC(q->nt, &q->tt);
    EMEMCPY(q->nt, mesh_get_tri(mesh), q->tt);
    snprintf(q->path, FILENAME_MAX, "%s/%s", DUMP_BASE, path);
    UC(efopen(q->path, "w", /**/ &f));
    UC(efclose(f));
    msg_print("DiagMesh: %s", q->path);
}

void diag_mesh_fin(DiagMesh *q) {
    EFREE(q->tt);
    EFREE(q);
}

void diag_mesh_apply(DiagMesh*, MPI_Comm, float, int, Particle*) {
    
}
