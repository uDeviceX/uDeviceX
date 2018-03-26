#define PATTERN "%s/%s/%05d.vtk"

void vtk_conf_ini(MeshRead *mesh, /**/ VTKConf **pq) {
    VTKConf *q;
    EMALLOC(1, &q);
    mesh_ini(mesh, &q->mesh);
    key_list_ini(&q->tri);
    key_list_ini(&q->vert);
    *pq = q;
}

void vtk_conf_fin(VTKConf *q) {
    key_list_fin(q->tri);
    key_list_fin(q->vert);
    mesh_fin(q->mesh);
    EFREE(q);
}

void vtk_conf_tri(VTKConf *q, const char *keys) {
    int n;
    n = key_list_size(q->tri);
    if (n > N_MAX)
        ERR("n=%d > N_MAX=%d", n, N_MAX);
    UC(key_list_push(q->tri, keys));
}

void vtk_conf_vert(VTKConf *q, const char *keys) {
    int n;
    n = key_list_size(q->tri);
    if (n > N_MAX)
        ERR("n=%d > N_MAX=%d", n, N_MAX);
    UC(key_list_push(q->vert, keys));
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
    /*
    for (i = 0; i < key_list_size(q->tri); i++)
        EMALLOC(q->D[i]);
    for (i = 0; i < key_list_size(q->vert); i++)
    EMALLOC(q->D[i]); */
    EMALLOC(nbuf, &q->D);

    q->nbuf = nbuf;
    q->nm       = UNSET;
    q->rr_set   = 0;
    UC(mesh_copy(mesh, /**/ &q->mesh));
    UC(key_list_copy(c->tri, &q->tri));
    UC(key_list_copy(c->vert, &q->vert));

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

void vtk_tri(VTK *q, int nm, const Scalars *sc, const char *keys) {
    int i, nt, n;
    if (!key_list_has(q->tri, keys)) {
        msg_print("unkown key '%s'", keys);
        key_list_log(q->tri);
        ERR("");
    }
    nt = mesh_nt(q->mesh);
    n = nm * nt;
    for (i = 0; i < n; i++)
        q->D[i] = scalars_get(sc, i);
    UC(key_list_mark(q->tri, keys));
}

void vtk_write(VTK *q, MPI_Comm comm, int id) {
    char path[FILENAME_MAX];
    int n, nm, nv, nt, nbuf;
    const int *tt;
    Out out;
    if (!key_list_marked(q->tri)) {
        msg_print("missing triangle data");
        key_list_log(q->tri);
        ERR("");
    }
    if (!key_list_marked(q->vert)) {
        msg_print("missing vertices data");
        key_list_log(q->vert);
        ERR("");
    }
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
    cell_header(&out, nm*nt);
    cell_data(&out, nm*nt, q->D, "area");

    UC(write_file_close(out.file));

    q->rr_set = 0;
    q->nm = UNSET;
    key_list_unmark(q->tri);
    key_list_unmark(q->vert);
}

void vtk_fin(VTK *q) {
    key_list_fin(q->tri);
    key_list_fin(q->vert);
    EFREE(q->dbuf);
    EFREE(q->ibuf);
    EFREE(q->D);
    EFREE(q);
}
