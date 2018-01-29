static void setup_from_pos(const Coords *coords, MPI_Comm comm, const char *cell, const char *ic, int nv, /**/
                           Particle *pp, int *pnc, int *pn, /* storage */ Particle *pp_hst) {
    int nc;
    nc = rbc_gen(coords, cell, ic, nv, pp_hst);
    if (nc) cH2D(pp, pp_hst, nv * nc);
    MC(m::Barrier(comm));
    *pnc = nc; *pn = nc * nv;
}

static void gen_ids(MPI_Comm comm, long nc, /**/ int *ii) {
    long i, i0 = 0;
    MC(m::Exscan(&nc, &i0, 1, MPI_LONG, MPI_SUM, comm));
    for (i = 0; i < nc; ++i)
        ii[i] = i + i0;
}

void rbc_gen_quants(const Coords *coords, MPI_Comm comm, const char *cell, const char *ic, /**/ RbcQuants *q) {
    int md, nt, nv;
    md = RBCmd;
    nt = RBCnt;
    nv = RBCnv;
    setup(md, nt, nv, cell, /**/ q);
    setup_from_pos(coords, comm, cell, ic, q->nv, /**/ q->pp, &q->nc, &q->n, /*w*/ q->pp_hst);
    if (rbc_ids)
        gen_ids(comm, q->nc, /**/ q->ii);
}
