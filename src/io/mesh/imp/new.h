static void ini(const int4 *tt, int nv, int nt, const char *directory, /**/ MeshWrite **pq) {
    int i;
    MeshWrite *q;
    UC(emalloc(sizeof(MeshWrite), (void**)&q));

    q->nv = nv; q->nt = nt; q->directory_exists = 0;
    strncpy(q->directory, directory, FILENAME_MAX);
    UC(emalloc(nt*sizeof(q->tt[0]), (void**)&q->tt));
    for (i = 0; i < nt; i++) q->tt[i] = tt[i];
    *pq = q;
}

void mesh_write_ini(const int4 *tt, int nv, int nt, const char *directory, /**/ MeshWrite **pq) {
    UC(ini(tt, nv, nt, directory, /**/ pq));
}

void mesh_write_ini_off(MeshRead *cell, const char *directory, /**/ MeshWrite **pq) {
    int nv, nt;
    const int4 *tt;
    nv = mesh_read_get_nv(cell);
    nt = mesh_read_get_nt(cell);
    tt = mesh_read_get_tri(cell);
    UC(ini(tt, nv, nt, directory, /**/ pq));
}

void mesh_write_fin(MeshWrite *q) {
    UC(efree(q->tt));
    UC(efree(q));
}

static void mkdir(const char *directory) {
    const char *fmt = "%s/%s";
    char directory0[FILENAME_MAX];
    if (sprintf(directory0, fmt, DUMP_BASE, directory) < 0)
        ERR("sprintf failed");
    UC(os_mkdir(directory0));
    msg_print("mkdir '%s'", directory0);
}
void mesh_write_dump(MeshWrite *q, MPI_Comm comm, const Coords *coords, int nc, const Particle *pp, int id) {
    const char *fmt = "%s/%s/%05d.ply";
    char path[FILENAME_MAX];
    if (!q->directory_exists) {
        q->directory_exists = 1;
        if (m::is_master(comm)) UC(mkdir(q->directory));
        msg_print("m::Barrier");
        MC(m::Barrier(comm));
    }
    if (sprintf(path, fmt, DUMP_BASE, q->directory, id) < 0)
        ERR("sprintf failed");
    UC(mesh_write(comm, coords, pp, q->tt, nc, q->nv, q->nt, path));
}
