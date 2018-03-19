static void ini(MPI_Comm comm, const int4 *tt, int nv, int nt, const char *directory, /**/ MeshWrite **pq) {
    int i;
    MeshWrite *q;
    EMALLOC(1, &q);

    q->nv = nv; q->nt = nt;
    cpy(q->directory, directory);
    UC(mkdir(comm, DUMP_BASE, directory));    
    EMALLOC(nt, &q->tt);
    for (i = 0; i < nt; i++)
        q->tt[i] = tt[i];
    q->shift_type = get_shift_type();
    *pq = q;
}

void mesh_write_ini(MPI_Comm comm, const int4 *tt, int nv, int nt, const char *directory, /**/ MeshWrite **pq) {
    UC(ini(comm, tt, nv, nt, directory, /**/ pq));
}

void mesh_write_ini_off(MPI_Comm comm, MeshRead *cell, const char *directory, /**/ MeshWrite **pq) {
    int nv, nt;
    const int4 *tt;
    nv = mesh_read_get_nv(cell);
    nt = mesh_read_get_nt(cell);
    tt = mesh_read_get_tri(cell);
    UC(ini(comm, tt, nv, nt, directory, /**/ pq));
}

void mesh_write_fin(MeshWrite *q) {
    EFREE(q->tt);
    EFREE(q);
}
