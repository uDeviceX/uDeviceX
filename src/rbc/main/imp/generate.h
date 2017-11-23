static void setup_from_pos(const char *cell, const char *r_state, int nv, /**/
                           Particle *pp, int *nc, int *n, /* storage */ Particle *pp_hst) {
    /* fills `pp' with RBCs for this processor */
    *nc = rbc::gen::main(cell, r_state, nv, pp_hst);
    if (*nc) cH2D(pp, pp_hst, nv * *nc);
    MC(m::Barrier(m::cart));
    *n = *nc * nv;
}

static void gen_ids(long nc, /**/ int *ii) {
    long i, i0 = 0;
    MC(m::Exscan(&nc, &i0, 1, MPI_LONG, MPI_SUM, m::cart));
    for (i = 0; i < nc; ++i)
        ii[i] = i + i0;
}

void gen_quants(const char *cell, const char *r_state, /**/ Quants *q) {
    int md, nt, nv;
    md = RBCmd;
    nt = RBCnt;
    nv = RBCnv;
    setup(md, nt, nv, cell, /**/
          q->shape.anti, q->shape.edg, &q->shape.totArea,
          q->tri_hst, q->tri, q->adj0, q->adj1);
    setup_from_pos(cell, r_state, q->nv, /**/ q->pp, &q->nc, &q->n, /*w*/ q->pp_hst);
    if (rbc_ids)
        gen_ids(q->nc, /**/ q->ii);
}
