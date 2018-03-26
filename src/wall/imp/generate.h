static void freeze0(MPI_Comm cart, int3 L, int maxn, const Sdf *qsdf, /*io*/ int *n, Particle *pp, /*o*/ int *w_n, Particle *dev, /*w*/ Particle *hst) {
    sdf_bulk_wall(qsdf, /*io*/ n, pp, /*o*/ w_n, hst); /* sort into bulk-frozen */
    msg_print("before exch: bulk/frozen : %d/%d", *n, *w_n);
    UC(wall_exch_pp(cart, L, maxn, /*io*/ hst, w_n));
    cH2D(dev, hst, *w_n);
    msg_print("after  exch: bulk/frozen : %d/%d", *n, *w_n);
}

static void freeze(MPI_Comm cart, int3 L, int maxn, const Sdf *qsdf, /*io*/ int *n, Particle *pp, /*o*/ int *w_n, Particle *dev) {
    Particle *hst;
    EMALLOC(maxn, &hst);
    UC(freeze0(cart, L, maxn, qsdf, /*io*/ n, pp, /*o*/ w_n, dev, /*w*/ hst));
    EFREE(hst);
}

static void gen_quants(MPI_Comm cart, int3 L, int maxn, const Sdf *qsdf, /**/ int *o_n, Particle *o_pp, int *w_n, float4 **w_pp) {
    Particle *frozen;
    CC(d::Malloc((void **) &frozen, maxn * sizeof(Particle)));
    UC(freeze(cart, L, maxn, qsdf, o_n, o_pp, w_n, frozen));
    msg_print("consolidating wall");
    CC(d::Malloc((void **) w_pp, *w_n * sizeof(float4)));
    KL(wall_dev::particle2float4, (k_cnf(*w_n)), (frozen, *w_n, /**/ *w_pp));
    
    CC(d::Free(frozen));
    dSync();
}

void wall_gen_quants(MPI_Comm cart, int maxn, const Sdf *sdf, /**/ int *n, Particle *pp, WallQuants *q) {
    int3 L = q->L;
    UC(gen_quants(cart, L, maxn, sdf, n, pp, &q->n, &q->pp));
}

static void build_cells(const int n, float4 *pp4, Clist *cells, ClistMap *mcells) {
    if (n == 0) return;

    Particle *pp, *pp0;
    CC(d::Malloc((void **) &pp,  n * sizeof(Particle)));
    CC(d::Malloc((void **) &pp0, n * sizeof(Particle)));

    KL(wall_dev::float42particle, (k_cnf(n)), (pp4, n, /**/ pp));
    UC(clist_build(n, n, pp, /**/ pp0, cells, mcells));
    KL(wall_dev::particle2float4, (k_cnf(n)), (pp0, n, /**/ pp4));

    CC(d::Free(pp));
    CC(d::Free(pp0));
}

static void gen_ticket(const int w_n, float4 *w_pp, Clist *cells, Texo<int> *texstart, Texo<float4> *texpp) {
    ClistMap *mcells;
    UC(clist_ini_map(w_n, 1, cells, /**/ &mcells));
    UC(build_cells(w_n, /**/ w_pp, cells, mcells));
    
    TE(texstart, cells->starts, cells->ncells);
    TE(texpp, w_pp, w_n);
    UC(clist_fin_map(mcells));
}

void wall_gen_ticket(const WallQuants *q, WallTicket *t) {
    UC(gen_ticket(q->n, q->pp, &t->cells, &t->texstart, &t->texpp));
}
