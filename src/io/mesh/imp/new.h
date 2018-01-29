static void ini(const int4 *tt, int nv, int nt, const char *directory, /**/ MeshWrite **pq) {
    int i;
    MeshWrite *q;
    UC(emalloc(sizeof(MeshWrite), (void**)&q));

    q->nv = nv; q->nt = nt;
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

void mesh_write_dump(MPI_Comm comm, const Coords *coord, int nc, const Particle *pp, MeshWrite *q, int id) {
}
