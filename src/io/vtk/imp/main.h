#define PATTERN "%s/%s/%05d.vtk"

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
    cpy(q->path, path);
    q->maxn = maxn;
    *pq = q;
}

void vtk_points(VTK *q, int nm, const Vectors *pos) {
    enum {X, Y, Z};
    float r[3];
    int nv, n, i, j;
    nv = mesh_nv(q->mesh);
    n = nv * nm;
    for (i = j = 0; i < n; i++) {
        UC(vectors_get(pos, i, r));
        q->rr[j++] = r[X];
        q->rr[j++] = r[Y];
        q->rr[j++] = r[Z];
    }
}

void vtk_write(VTK *q, MPI_Comm comm, int id) {
    char path[FILENAME_MAX];
    WriteFile *f;
    if (snprintf(path, FILENAME_MAX, PATTERN, DUMP_BASE, q->path, id) < 0)
        ERR("snprintf failed");

    write_file_open(comm, path, &f);
    write_file_close(f);
}

void vtk_fin(VTK *q) {
    EFREE(q->rr);
    EFREE(q);
}
