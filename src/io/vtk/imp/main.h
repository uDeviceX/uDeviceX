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

void vtk_ini(MPI_Comm comm, int maxn, char const *path, VTKConf *c, /**/ VTK **pq) {
    VTK *q;
    int nbuf;
    Mesh *mesh;
    EMALLOC(1, &q);
    mesh = c->mesh;
    nbuf = get_nbuf(maxn, mesh_nv(mesh), mesh_nt(mesh), mesh_ne(mesh));
    EMALLOC(nbuf, &q->dbuf);
    EMALLOC(nbuf, &q->ibuf);
    UC(mkdir(comm, DUMP_BASE, path));
    cpy(q->path, path);
    q->nbuf = nbuf;
    q->nm       = UNSET;
    q->rr_set   = 0;
    UC(mesh_copy(mesh, /**/ &q->mesh));

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
        q->dbuf[j++] = r[X];
        q->dbuf[j++] = r[Y];
        q->dbuf[j++] = r[Z];
    }

    q->nm = nm;
    q->rr_set = 1;
}

void vtk_write(VTK *q, MPI_Comm comm, int id) {
    char path[FILENAME_MAX];
    int n, nm, nv, nt, nbuf;
    const int *tt;
    Out out;
    if (!q->rr_set) ERR("points are unset");

    if (snprintf(path, FILENAME_MAX, PATTERN, DUMP_BASE, q->path, id) < 0)
        ERR("snprintf failed");
    out.comm = comm;
    UC(write_file_open(comm, path, &out.file));

    nm = q->nm; nbuf = q->nbuf; nv = mesh_nv(q->mesh);
    nt = mesh_nt(q->mesh); tt = mesh_tt(q->mesh);

    n = nm * nv;
    if (3 * n > nbuf)
        ERR("3*n=%d > nbuf=%d", n, nbuf);
    if (4 * nm * nt > nbuf)
        ERR("nm=%d * nt=%d > nbuf=%d", nm, nt, nbuf);
    header(&out);
    points(&out, n, q->dbuf);
    tri(&out, nm, nv, nt, tt, q->ibuf);
    UC(write_file_close(out.file));

    q->rr_set = 0;
    q->nm = UNSET;
}

void vtk_fin(VTK *q) {
    EFREE(q->dbuf);
    EFREE(q->ibuf);
    EFREE(q);
}
