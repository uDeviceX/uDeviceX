#define PP  ".pp"
#define IDS ".ids"

static void gen_code(const char *name, const char *ext, char *code) {
    strcpy(code, name);
    strcat(code, ext);
}

static void setup_from_strt(MPI_Comm comm, const char *base, const char *name, int nv, int id, /**/ Particle *pp, int *nc, int *n, /*w*/ Particle *pp_hst) {
    char code[FILENAME_MAX];
    gen_code(name, PP, code);
    restart_read_pp(comm, base, code, id, n, pp_hst);
    *nc = *n / nv;

    if (*n) cH2D(pp, pp_hst, *n);
}

static void ids_from_strt(MPI_Comm comm, const char *base, const char *name, int id, /**/ int *ii) {
    int nc;
    char code[FILENAME_MAX];
    gen_code(name, IDS, code);
    restart_read_ii(comm, base, code, id, &nc, ii);
}

void rbc_strt_quants(MPI_Comm comm, const MeshRead *off, const char *base, const char *name, int id, RbcQuants *q) {
    int nv;
    nv = mesh_read_get_nv(off);
    setup_from_strt(comm, base, name, nv, id, /**/ q->pp, &q->nc, &q->n, /*w*/ q->pp_hst);
    if (q->ids) ids_from_strt(comm, base, name, id, /**/ q->ii);
}

static void strt_dump(MPI_Comm comm, const char *base, const char *name, int id, int n, const Particle *pp, /*w*/ Particle *pp_hst) {
    char code[FILENAME_MAX];
    gen_code(name, PP, code);
    if (n) cD2H(pp_hst, pp, n);
    restart_write_pp(comm, base, code, id, n, pp_hst);
}

static void strt_dump_ii(MPI_Comm comm, const char *base, const char *name, int id, int nc, const int *ii) {
    char code[FILENAME_MAX];
    gen_code(name, IDS, code);
    restart_write_ii(comm, base, code, id, nc, ii);
}

void rbc_strt_dump(MPI_Comm comm, const char *base, const char *name, int id, const RbcQuants *q) {
    strt_dump(comm, base, name, id, q->n, q->pp, /*w*/ q->pp_hst);
    if (q->ids)
        strt_dump_ii(comm, base, name, id, q->nc, q->ii);
}

#undef PP
#undef IDS
