static void read(int maxn, /**/ float4 *pp, int *n) {
    Particle *pphst, *ppdev;
    size_t sz = maxn * sizeof(Particle);
    UC(emalloc(sz, (void**) &pphst));

    restart::read_pp("wall", restart::TEMPL, /**/ pphst, n);

    if (*n) {
        CC(d::Malloc((void **) &ppdev, sz));
        cH2D(ppdev, pphst, *n);
        KL(dev::particle2float4, (k_cnf(*n)), (ppdev, *n, /**/ pp));
        CC(d::Free(ppdev));
    }
    free(pphst);
}

static void write(int n, const float4 *pp) {
    Particle *pphst, *ppdev;
    size_t sz = n * sizeof(Particle);

    UC(emalloc(sz, (void**) &pphst));
    if (n) {
        CC(d::Malloc((void **) &ppdev, n * sizeof(Particle)));
        KL(dev::float42particle , (k_cnf(n)), (pp, n, /**/ ppdev));
        cD2H(pphst, ppdev, n);
        CC(d::Free(ppdev));
    }
    restart::write_pp("wall", restart::TEMPL, /**/ pphst, n);

    free(pphst);
}

static void strt_quants(int maxn, int *w_n, float4 **w_pp) {
    float4 * pptmp;
    CC(d::Malloc((void **) &pptmp, maxn * sizeof(float4)));
    read(maxn, pptmp, w_n);

    if (*w_n) {
        CC(d::Malloc((void **) w_pp, *w_n * sizeof(float4)));
        cD2D(*w_pp, pptmp, *w_n);
    }
    CC(d::Free(pptmp));
}

void strt_quants(int maxn, Quants *q) {
    strt_quants(maxn, &q->n, &q->pp);
}

void strt_dump_templ(const Quants q) {
    write(q.n, q.pp);
}
