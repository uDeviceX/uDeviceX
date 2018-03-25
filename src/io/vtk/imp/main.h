void vtk_conf_ini(MeshRead *mesh, /**/ VTKConf **pq) {
    VTKConf *q;
    EMALLOC(1, &q);
    mesh_ini(mesh, &q->mesh);
    key_list_ini(&q->tri);
    *pq = q;
}

void vtk_conf_fin(VTKConf *q) {
    key_list_fin(q->tri);
    mesh_fin(q->mesh);
    EFREE(q);
}

void vtk_conf_tri(VTKConf *q, const char *keys) {
    UC(key_list_push(q->tri, keys));
}

void vtk_ini(int maxn, char const *path, VTKConf *c, /**/ VTK **pq) {
    VTK *q;
    EMALLOC(1, &q);
    EMALLOC(3*maxn, &q->rr);
    mesh_copy(c->mesh, /**/ &q->mesh);
    UC(mkdir(DUMP_BASE, path));
    q->maxn = maxn;
    *pq = q;
}

void vtk_points(VTK *q, int nm, const Vectors *pos) {
    int n, i;
    for (i = 0; i < n; i++) {
    }
}

void vtk_fin(VTK *q) {
    EFREE(q->rr);
    EFREE(q);
}
