#define CODE "rbc"
#define PP  CODE ".pp"
#define IDS CODE ".ids"

static void setup_from_strt(MPI_Comm comm, const char *base, int nv, int id, /**/ Particle *pp, int *nc, int *n, /*w*/ Particle *pp_hst) {
    restart_read_pp(comm, base, PP, id, n, pp_hst);
    *nc = *n / nv;

    if (*n) cH2D(pp, pp_hst, *n);
}

static void ids_from_strt(MPI_Comm comm, const char *base, int id, /**/ int *ii) {
    int nc;
    restart_read_ii(comm, base, IDS, id, &nc, ii);
}

void rbc_strt_quants(MPI_Comm comm, const char *base, MeshRead *off, int id, RbcQuants *q) {
    int nv;
    nv = mesh_read_get_nv(off);
    setup_from_strt(comm, base, nv, id, /**/ q->pp, &q->nc, &q->n, /*w*/ q->pp_hst);
    if (q->ids) ids_from_strt(comm, base, id, /**/ q->ii);
}

static void strt_dump(MPI_Comm comm, const char *base, int id, int n, const Particle *pp, /*w*/ Particle *pp_hst) {
    if (n) cD2H(pp_hst, pp, n);
    restart_write_pp(comm, base, PP, id, n, pp_hst);
}

static void strt_dump_ii(MPI_Comm comm, const char *base, int id, int nc, const int *ii) {
    restart_write_ii(comm, base, IDS, id, nc, ii);
}

void rbc_strt_dump(MPI_Comm comm, const char *base, int id, const RbcQuants *q) {
    strt_dump(comm, base, id, q->n, q->pp, /*w*/ q->pp_hst);
    if (q->ids)
        strt_dump_ii(comm, base, id, q->nc, q->ii);
}

#undef CODE
#undef PP
#undef IDS
