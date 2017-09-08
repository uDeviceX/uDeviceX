typedef const sdf::tex3Dca<float> TexSDF_t;

static void freeze0(TexSDF_t texsdf, /*io*/ Particle *pp, int *n, /*o*/ Particle *dev, int *w_n, /*w*/ Particle *hst) {
    sdf::bulk_wall(texsdf, /*io*/ pp, n, /*o*/ hst, w_n); /* sort into bulk-frozen */
    MSG("before exch: bulk/frozen : %d/%d", *n, *w_n);
    exch(/*io*/ hst, w_n);
    cH2D(dev, hst, *w_n);
    MSG("after  exch: bulk/frozen : %d/%d", *n, *w_n);
}

static void freeze(TexSDF_t texsdf, /*io*/ Particle *pp, int *n, /*o*/ Particle *dev, int *w_n) {
    Particle *hst;
    hst = (Particle*)malloc(MAX_PART_NUM*sizeof(Particle));
    freeze0(texsdf, /*io*/ pp, n, /*o*/ dev, w_n, /*w*/ hst);
    free(hst);
}

void build_cells(const int n, float4 *pp4, clist::Clist *cells) {
    if (n == 0) return;

    Particle *pp;
    CC(cudaMalloc(&pp, n * sizeof(Particle)));

    KL(dev::float42particle, (k_cnf(n)), (pp4, n, /**/ pp));
    cells->build(pp, n);
    KL(dev::particle2float4, (k_cnf(n)), (pp, n, /**/ pp4));

    CC(cudaFree(pp));
}

void gen_quants(TexSDF_t texsdf, /**/ int *o_n, Particle *o_pp, int *w_n, float4 **w_pp) {
    Particle *frozen;
    CC(cudaMalloc(&frozen, sizeof(Particle) * MAX_PART_NUM));
    freeze(texsdf, o_pp, o_n, frozen, w_n);
    MSG("consolidating wall");
    CC(cudaMalloc(w_pp, *w_n * sizeof(float4)));
    KL(dev::particle2float4, (k_cnf(*w_n)), (frozen, *w_n, /**/ *w_pp));
    
    CC(cudaFree(frozen));
    dSync();
}

void strt_quants(int *w_n, float4 **w_pp) {
    float4 * pptmp; CC(cudaMalloc(&pptmp, MAX_PART_NUM * sizeof(float4)));
    strt::read(pptmp, w_n);

    if (*w_n) {
        CC(cudaMalloc(w_pp, *w_n * sizeof(float4)));
        cD2D(*w_pp, pptmp, *w_n);
    }
    CC(cudaFree(pptmp));
}

void gen_ticket(const int w_n, float4 *w_pp, clist::Clist *cells, Texo<int> *texstart, Texo<float4> *texpp) {

    build_cells(w_n, /**/ w_pp, cells);
    
    TE(texstart, cells->start, cells->ncells);
    TE(texpp, w_pp, w_n);
}

void pair(TexSDF_t texsdf, const int type, hforces::Cloud cloud, const int n, const Texo<int> texstart,
                  const Texo<float4> texpp, const int w_n, /**/ rnd::KISS *rnd, Force *ff) {
    KL(dev::pair,
       (k_cnf(3*n)),
       (texsdf, cloud, n, w_n, (float *)ff, rnd->get_float(), type, texstart, texpp));
}

void strt_dump_templ(const int n, const float4 *pp) {
    strt::write(pp, n);
}
