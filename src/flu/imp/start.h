#define CODE "flu"
#define PP  CODE ".pp"
#define COL CODE ".cc"
#define IDS CODE ".ii"

static int strt_pp(MPI_Comm comm, const int id, Particle *dev, /*w*/ Particle *hst) {
    int n;
    restart_read_pp(comm, PP, id, &n, hst);
    if (n) cH2D(dev, hst, n);
    return n;
}

static int strt_ii(MPI_Comm comm, const char *code, const int id, int *dev, /*w*/ int *hst) {
    int n;
    restart_read_ii(comm, code, id, &n, hst);
    if (n) cH2D(dev, hst, n);
    return n;
}

void flu_strt_quants(MPI_Comm comm, const int id, FluQuants *q) {
    q->n =         strt_pp(comm,     id, /**/ q->pp, /* w */ q->pp_hst);
    if (q->ids)    strt_ii(comm, IDS, id, /**/ q->ii, /* w */ q->ii_hst);
    if (q->colors) strt_ii(comm, COL, id, /**/ q->cc, /* w */ q->cc_hst);
}

static void strt_dump_pp(MPI_Comm comm, const int id, const int n, const Particle *dev, Particle *hst) {
    if (n) cD2H(hst, dev, n);
    restart_write_pp(comm, PP, id, n, hst);
}

static void strt_dump_ii(MPI_Comm comm, const char *code, const int id, const int n, const int *dev, int *hst) {
    if (n) cD2H(hst, dev, n);
    restart_write_ii(comm, code, id, n, hst);
}

void flu_strt_dump(MPI_Comm comm, const int id, const FluQuants *q) {
    strt_dump_pp(comm, id, q->n, q->pp, /* w */ q->pp_hst);
    if (q->ids)    strt_dump_ii(comm, IDS, id, q->n, q->ii, /* w */ q->ii_hst);
    if (q->colors) strt_dump_ii(comm, COL, id, q->n, q->cc, /* w */ q->cc_hst);
}

#undef CODE
#undef COl
#undef IDS
#undef PP
