static void freeze0(MPI_Comm cart, int maxn, Sdf *qsdf, /*io*/ Particle *pp, int *n, /*o*/ Particle *dev, int *w_n, /*w*/ Particle *hst) {
    bulk_wall(qsdf, /*io*/ pp, n, /*o*/ hst, w_n); /* sort into bulk-frozen */
    msg_print("before exch: bulk/frozen : %d/%d", *n, *w_n);
    UC(exch(cart, maxn, /*io*/ hst, w_n));
    cH2D(dev, hst, *w_n);
    msg_print("after  exch: bulk/frozen : %d/%d", *n, *w_n);
}

static void freeze(MPI_Comm cart, int maxn, Sdf *qsdf, /*io*/ Particle *pp, int *n, /*o*/ Particle *dev, int *w_n) {
    Particle *hst;
    UC(emalloc(maxn * sizeof(Particle), (void**) &hst));
    UC(freeze0(cart, maxn, qsdf, /*io*/ pp, n, /*o*/ dev, w_n, /*w*/ hst));
    free(hst);
}

static void gen_quants(MPI_Comm cart, int maxn, Sdf *qsdf, /**/ int *o_n, Particle *o_pp, int *w_n, float4 **w_pp) {
    Particle *frozen;
    CC(d::Malloc((void **) &frozen, maxn * sizeof(Particle)));
    UC(freeze(cart, maxn, qsdf, o_pp, o_n, frozen, w_n));
    msg_print("consolidating wall");
    CC(d::Malloc((void **) w_pp, *w_n * sizeof(float4)));
    KL(dev::particle2float4, (k_cnf(*w_n)), (frozen, *w_n, /**/ *w_pp));
    
    CC(d::Free(frozen));
    dSync();
}

void gen_quants(MPI_Comm cart, int maxn, Sdf *sdf, /**/ int *n, Particle* pp, Quants *q) {
    UC(gen_quants(cart, maxn, sdf, n, pp, &q->n, &q->pp));
}

static void build_cells(const int n, float4 *pp4, clist::Clist *cells, clist::Map *mcells) {
    if (n == 0) return;

    Particle *pp, *pp0;
    CC(d::Malloc((void **) &pp,  n * sizeof(Particle)));
    CC(d::Malloc((void **) &pp0, n * sizeof(Particle)));

    KL(dev::float42particle, (k_cnf(n)), (pp4, n, /**/ pp));
    UC(clist::build(n, n, pp, /**/ pp0, cells, mcells));
    KL(dev::particle2float4, (k_cnf(n)), (pp0, n, /**/ pp4));

    CC(d::Free(pp));
    CC(d::Free(pp0));
}

static void gen_ticket(const int w_n, float4 *w_pp, clist::Clist *cells, Texo<int> *texstart, Texo<float4> *texpp) {
    clist::Map mcells;
    UC(ini_map(w_n, 1, cells, /**/ &mcells));
    UC(build_cells(w_n, /**/ w_pp, cells, &mcells));
    
    TE(texstart, cells->starts, cells->ncells);
    TE(texpp, w_pp, w_n);
    UC(fin_map(&mcells));
}

void gen_ticket(const Quants q, Ticket *t) {
    UC(gen_ticket(q.n, q.pp, &t->cells, &t->texstart, &t->texpp));
}
