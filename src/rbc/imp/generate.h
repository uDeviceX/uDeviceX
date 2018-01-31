static void setup_from_pos(const Coords *coords, MPI_Comm comm, const float *vv, const char *ic, int nv, /**/
                           Particle *pp, int *pnc, int *pn, /* storage */ Particle *pp_hst) {
    int nc;
    nc = rbc_gen(coords, vv, ic, nv, /**/ pp_hst);
    if (nc) cH2D(pp, pp_hst, nv * nc);
    MC(m::Barrier(comm));
    *pnc = nc; *pn = nc * nv;
}

static void gen_ids(MPI_Comm comm, long nc, /**/ int *ii) {
    long i, i0 = 0, count = 1;
    MC(m::Exscan(&nc, &i0, count, MPI_LONG, MPI_SUM, comm));
    for (i = 0; i < nc; ++i)
        ii[i] = i + i0;
}

void rbc_gen_quants(const Coords *coords, MPI_Comm comm, OffRead *off, const char *ic, /**/ RbcQuants *q) {
    int md, nt, nv;
    const float *vv;
    md = RBCmd; nt = off_get_nt(off); nv = off_get_nv(off); vv = off_get_vert(off);
    setup_from_pos(coords, comm, vv, ic, q->nv, /**/ q->pp, &q->nc, &q->n, /*w*/ q->pp_hst);
    if (rbc_ids) gen_ids(comm, q->nc, /**/ q->ii);
}
