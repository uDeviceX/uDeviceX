static void gen(const char *cell, const char *ic, int nv, /**/ int *pnc, Particle *dev) {
    int nc, sz;
    Particle *hst;
    sz = sizeof(Particle)*MAX_CELL_NUM*nv;
    UC(emalloc(sz, (void**)&hst));

    nc = rbc::gen::main(cell, ic, nv, /**/ hst);
    MSG("nc = %d, nv = %d", nc, nv);
    if (nc) cH2D(dev, hst, nc*nv);

    UC(efree(hst));
    *pnc = nc;
}

static void setup_from_pos(MPI_Comm comm, const char *cell, const char *ic, int nv, /**/
                           Particle *pp, int *pnc, int *pn) {
    int nc;
    UC(gen(cell, ic, nv, /**/ &nc, pp));
    MC(m::Barrier(comm));
    *pnc = nc; *pn = nc * nv;
}

static void gen_ids(MPI_Comm comm, long nc, /**/ int *ii) {
    long i, i0 = 0;
    MC(m::Exscan(&nc, &i0, 1, MPI_LONG, MPI_SUM, comm));
    for (i = 0; i < nc; ++i)
        ii[i] = i + i0;
}

void gen_quants(MPI_Comm comm, const char *cell, const char *ic, /**/ Quants *q) {
    int md, nt, nv;
    md = RBCmd;
    nt = RBCnt;
    nv = RBCnv;
    UC(setup(md, nt, nv, cell, /**/
             q->shape.anti, q->shape.edg, &q->shape.totArea,
             q->tri_hst, q->tri, q->adj0, q->adj1));
    UC(setup_from_pos(comm, cell, ic, q->nv, /**/ q->pp, &q->nc, &q->n));
    if (rbc_ids)
        gen_ids(comm, q->nc, /**/ q->ii);
}
