void diag_mesh_ini(const char *path, MeshRead *mesh, DiagMesh **pq) {
    FILE *f;
    DiagMesh *q;
    EMALLOC(1, &q);
    *pq = q;
    
    q->nv = mesh_get_nv(mesh);
    q->nt = mesh_get_nt(mesh);
    
    snprintf(q->path, FILENAME_MAX, "%s/%s", DUMP_BASE, path);
    UC(efopen(q->path, "w", /**/ &f));
    UC(efclose(f));
    msg_print("DiagMesh: %s", q->path);
}

void diag_mesh_fin(DiagMesh*) {

}

void diag_mesh_apply(DiagMesh*, MPI_Comm, float, int, Particle*) {
}
