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

void mesh_write_ini_off(OffRead *cell, const char *directory, /**/ MeshWrite **pq) {
    int nv, nt;
    const int4 *tt;
    nv = off_get_nv(cell);
    nt = off_get_nt(cell);
    tt = off_get_tri(cell);
    UC(ini(tt, nv, nt, directory, /**/ pq));
}

void mesh_write_fin(MeshWrite *q) {
    UC(efree(q->tt));
    UC(efree(q));
}

static void mkdir(const char* directory) {
    char directory0[FILENAME_MAX];
    if (sprintf(directory0, "%s/%s", DUMP_BASE, directory) < 0)
        ERR("sprintf failed");
    UC(os_mkdir(directory0));
    msg_print("mkdir '%s'", directory0);
}
void mesh_write_dump(MeshWrite *q, MPI_Comm comm, const Coords *coord, int nc, const Particle *pp, int id) {
    if (!q->directory_exists) {
        q->directory_exists = 1;
        UC(mkdir(q->directory));
    }
}
