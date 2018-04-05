#define CODE "wall"
#define PP CODE ".pp"

static void read(MPI_Comm comm, const char *base, int maxn, /**/ float4 *pp, int *n) {
    Particle *pphst, *ppdev;
    size_t sz = maxn * sizeof(Particle);
    EMALLOC(maxn, &pphst);

    restart_read_pp(comm, base, PP, RESTART_TEMPL, n, pphst);

    if (*n) {
        CC(d::Malloc((void **) &ppdev, sz));
        cH2D(ppdev, pphst, *n);
        KL(wall_dev::particle2float4, (k_cnf(*n)), (ppdev, *n, /**/ pp));
        CC(d::Free(ppdev));
    }
    EFREE(pphst);
}

static void write(MPI_Comm comm, const char *base, int n, const float4 *pp) {
    Particle *pphst, *ppdev;

    EMALLOC(n, &pphst);
    if (n) {
        CC(d::Malloc((void **) &ppdev, n * sizeof(Particle)));
        KL(wall_dev::float42particle , (k_cnf(n)), (pp, n, /**/ ppdev));
        cD2H(pphst, ppdev, n);
        CC(d::Free(ppdev));
    }
    restart_write_pp(comm, base, PP, RESTART_TEMPL, n, pphst);

    EFREE(pphst);
}

static void strt_quants(MPI_Comm comm, const char *base, int maxn, int *w_n, float4 **w_pp) {
    float4 * pptmp;
    CC(d::Malloc((void **) &pptmp, maxn * sizeof(float4)));
    read(comm, base, maxn, pptmp, w_n);

    if (*w_n) {
        CC(d::Malloc((void **) w_pp, *w_n * sizeof(float4)));
        cD2D(*w_pp, pptmp, *w_n);
    }
    CC(d::Free(pptmp));
}

void wall_strt_quants(MPI_Comm comm, const char *base, int maxn, WallQuants *q) {
    strt_quants(comm, base, maxn, &q->n, &q->pp);
}

void wall_strt_dump_templ(MPI_Comm comm, const char *base, const WallQuants *q) {
    write(comm, base, q->n, q->pp);
}

#undef PP
#undef CODE
