static void freeze0(sdf::Tex_t texsdf, /*io*/ Particle *pp, int *n, /*o*/ Particle *dev, int *w_n, /*w*/ Particle *hst) {
    sdf::bulk_wall(texsdf, /*io*/ pp, n, /*o*/ hst, w_n); /* sort into bulk-frozen */
    MSG("before exch: bulk/frozen : %d/%d", *n, *w_n);
    exch(/*io*/ hst, w_n);
    cH2D(dev, hst, *w_n);
    MSG("after  exch: bulk/frozen : %d/%d", *n, *w_n);
}

static void freeze(sdf::Tex_t texsdf, /*io*/ Particle *pp, int *n, /*o*/ Particle *dev, int *w_n) {
    Particle *hst;
    hst = (Particle*)malloc(MAX_PART_NUM*sizeof(Particle));
    freeze0(texsdf, /*io*/ pp, n, /*o*/ dev, w_n, /*w*/ hst);
    free(hst);
}

void build_cells(const int n, float4 *pp4, clist::Clist *cells, clist::Map *tcells) {
    if (n == 0) return;

    Particle *pp, *pp0;
    CC(d::Malloc((void **) &pp,  n * sizeof(Particle)));
    CC(d::Malloc((void **) &pp0, n * sizeof(Particle)));

    KL(dev::float42particle, (k_cnf(n)), (pp4, n, /**/ pp));
    clist::build(n, n, pp, /**/ pp0, cells, tcells);
    KL(dev::particle2float4, (k_cnf(n)), (pp0, n, /**/ pp4));

    CC(d::Free(pp));
    CC(d::Free(pp0));
}

void gen_quants(sdf::Tex_t texsdf, /**/ int *o_n, Particle *o_pp, int *w_n, float4 **w_pp) {
    Particle *frozen;
    CC(d::Malloc((void **) &frozen, sizeof(Particle) * MAX_PART_NUM));
    freeze(texsdf, o_pp, o_n, frozen, w_n);
    MSG("consolidating wall");
    CC(d::Malloc((void **) w_pp, *w_n * sizeof(float4)));
    KL(dev::particle2float4, (k_cnf(*w_n)), (frozen, *w_n, /**/ *w_pp));
    
    CC(d::Free(frozen));
    dSync();
}

void strt_quants(int *w_n, float4 **w_pp) {
    float4 * pptmp;
    CC(d::Malloc((void **) &pptmp, MAX_PART_NUM * sizeof(float4)));
    strt::read(pptmp, w_n);

    if (*w_n) {
        CC(d::Malloc((void **) w_pp, *w_n * sizeof(float4)));
        cD2D(*w_pp, pptmp, *w_n);
    }
    CC(d::Free(pptmp));
}

void gen_ticket(const int w_n, float4 *w_pp, clist::Clist *cells, clist::Map *tcells, Texo<int> *texstart, Texo<float4> *texpp) {

    build_cells(w_n, /**/ w_pp, cells, tcells);
    
    TE(texstart, cells->starts, cells->ncells);
    TE(texpp, w_pp, w_n);
}

void strt_dump_templ(const int n, const float4 *pp) {
    strt::write(pp, n);
}
