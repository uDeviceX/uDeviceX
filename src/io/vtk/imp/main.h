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
    UC(mesh_copy(c->mesh, /**/ &q->mesh));
    UC(mkdir(DUMP_BASE, path));
    cpy(q->path, path);

    q->maxn = maxn;
    q->nm       = UNSET;
    q->rr_set   = 0;

    *pq = q;
}

void vtk_points(VTK *q, int nm, const Vectors *pos) {
    enum {X, Y, Z};
    float r[3];
    int nv, n, i, j;
    if (q->nm != UNSET && q->nm != nm)
        ERR("q->nm=%d  != nm=%d", q->nm, nm);

    nv = mesh_nv(q->mesh);
    n = nv * nm;
    for (i = j = 0; i < n; i++) {
        UC(vectors_get(pos, i, r));
        q->rr[j++] = r[X];
        q->rr[j++] = r[Y];
        q->rr[j++] = r[Z];
    }

    q->nm = nm;
    q->rr_set = 1;
}

void vtk_write(VTK *q, MPI_Comm comm, int id) {
    char path[FILENAME_MAX];
    int n;
    Out out;
    if (!q->rr_set) ERR("points are unset");

    if (snprintf(path, FILENAME_MAX, PATTERN, DUMP_BASE, q->path, id) < 0)
        ERR("snprintf failed");
    out.comm = comm;
    UC(write_file_open(comm, path, &out.f));

    n = q->nm * mesh_nv(q->mesh);
    header(&out);
    points(&out, n, q->rr);

    UC(write_file_close(out.f));

    q->rr_set = 0;
    q->nm = UNSET;
}

void vtk_fin(VTK *q) {
    EFREE(q->rr);
    EFREE(q);
}
