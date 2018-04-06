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

void vtk_ini(MPI_Comm comm, int maxm, char const *path, VTKConf *c, /**/ VTK **pq) {
    int i;
    VTK *q;
    int nbuf, nk;
    Mesh *mesh;
    EMALLOC(1, &q);
    mesh = c->mesh;
    nbuf = get_nbuf(maxm, mesh_nv(mesh), mesh_nt(mesh), mesh_ne(mesh));
    EMALLOC(nbuf, &q->dbuf);
    EMALLOC(nbuf, &q->ibuf);
    UC(mkdir(comm, DUMP_BASE, path));
    cpy(q->path, path);
    UC(nk = key_list_size(c->tri));
    for (i = 0; i < nk; i++) EMALLOC(nbuf, &q->TRI[i]);
    UC(nk = key_list_size(c->vert));
    for (i = 0; i < nk; i++) EMALLOC(nbuf, &q->VERT[i]);

    q->stamp = MAGIC;
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
    int i, j, nt, n;
    if (!key_list_has(q->tri, keys)) {
        msg_print("unkown key '%s'", keys);
        key_list_log(q->tri);
        ERR("");
    }
    j = key_list_offset(q->tri, keys);
    nt = mesh_nt(q->mesh);
    n = nm * nt;
    for (i = 0; i < n; i++)
        q->TRI[j][i] = scalars_get(sc, i);
    UC(key_list_mark(q->tri, keys));
}

void vtk_vert(VTK *q, int nm, const Scalars *sc, const char *keys) {
    int i, j, nv, n;
    if (!key_list_has(q->vert, keys)) {
        msg_print("unkown key '%s'", keys);
        key_list_log(q->vert);
        ERR("wrong vtk_vert call");
    }
    j = key_list_offset(q->vert, keys);
    nv = mesh_nv(q->mesh);
    n = nm * nv;
    for (i = 0; i < n; i++)
        UC(q->VERT[j][i] = scalars_get(sc, i));
    UC(key_list_mark(q->vert, keys));
}

void vtk_write(VTK *q, MPI_Comm comm, int id) {
    char path[FILENAME_MAX];
    int i, nk, n, nm, nv, nt, nbuf;
    const int *tt;
    const char *keys;
    Out out;
    if (q->stamp != MAGIC)
        ERR("VTK is not initialized");

    if (!key_list_marked(q->tri)) {
        msg_print("missing triangle data");
        key_list_log(q->tri);
        ERR("vtk_write is called is not enough data");
    }
    if (!key_list_marked(q->vert)) {
        msg_print("missing vertices data");
        key_list_log(q->vert);
        ERR("vtk_write is called is not enough data");
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

    UC(nk = key_list_size(q->tri));
    if (nk > 0) cell_header(&out, nm*nt);
    for (i = 0; i < nk; i++) {
        keys = key_list_name(q->tri, i);
        cell_data(&out, nm*nt, q->TRI[i], keys);
    }

    UC(nk = key_list_size(q->vert));
    if (nk > 0) point_header(&out, n);
    for (i = 0; i < nk; i++) {
        keys = key_list_name(q->vert, i);
        point_data(&out, n, q->VERT[i], keys);
    }

    UC(write_file_close(out.file));

    q->rr_set = 0;
    q->nm = UNSET;
    key_list_unmark(q->tri);
    key_list_unmark(q->vert);
}

void vtk_fin(VTK *q) {
    int i;
    if (q->stamp != MAGIC)
        ERR("VTK is not initialized");
    EFREE(q->dbuf);
    EFREE(q->ibuf);
    for (i = 0; i < key_list_size(q->tri); i++)
        EFREE(q->TRI[i]);
    for (i = 0; i < key_list_size(q->vert); i++)
        EFREE(q->VERT[i]);
    key_list_fin(q->tri);
    key_list_fin(q->vert);
    EFREE(q);
}
