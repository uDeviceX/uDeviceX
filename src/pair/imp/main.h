void pair_ini(PairParams **par) {
    PairParams *p;
    UC(emalloc(sizeof(PairParams), (void **) par));
    p = *par;

    p->ncol = 0;
}

void pair_fin(PairParams *p) {
    UC(efree(p));
}

void pair_set_lj(float sigma, float eps, PairParams *p) {
    p->lj_s = sigma;
    p->lj_e = eps;
}

static int get_npar(int ncol) { return (ncol * (ncol+1)) / 2; }

void pair_set_dpd(int ncol, const float a[], const float g[], const float s[], PairParams *p) {
    int npar = get_npar(ncol);
    size_t sz = sizeof(float) * npar;

    if (ncol <= 0)      ERR("need at least one color");
    if (ncol > MAX_COL) ERR("too many colors");
    
    p->ncol = ncol;
    memcpy(p->a, a, sz);
    memcpy(p->g, g, sz);
    memcpy(p->s, s, sz);
}

