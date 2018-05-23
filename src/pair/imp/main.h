void pair_ini(PairParams **par) {
    PairParams *p;
    EMALLOC(1, par);
    p = *par;

    p->ncolors = 0;
    p->dpd = p->lj = p->adhesion = false;
}

void pair_fin(PairParams *p) {
    EFREE(p);
}

void pair_set_lj(float sigma, float eps, PairParams *p) {
    p->lj = true;
    p->ljs = sigma;
    p->lje = eps;
}

static int get_npar(int ncol) { return (ncol * (ncol+1)) / 2; }

void pair_set_dpd(int ncol, const float a[], const float g[], float spow, PairParams *p) {
    int npar = get_npar(ncol);
    size_t sz = sizeof(float) * npar;

    if (ncol <= 0)      ERR("need at least one color (given %d)\n", ncol);
    if (ncol > N_COLOR) ERR("too many colors (given %d)\n", ncol);

    p->dpd = true;
    p->ncolors = ncol;
    memcpy(p->a, a, sz);
    memcpy(p->g, g, sz);
    p->spow = spow;
}

void pair_set_adhesion(float k1, float k2, PairParams *p) {
    p->adhesion = true;
    p->k1 = k1;
    p->k2 = k2;
}

static void check_dpd(const PairParams *p) {
    if (!p->dpd) ERR("Must have dpd parameters");
}

static void check_lj(const PairParams *p) {
    if (!p->lj) ERR("Must have lj parameters");
}

static void check_adhesion(const PairParams *p) {
    if (!p->adhesion) ERR("Must have adhesion parameters");
}

void pair_compute_dpd_sigma(float kBT, float dt, PairParams *p) {
    int i, npar, ncol;
    UC(check_dpd(p));
    ncol = p->ncolors;
    npar = get_npar(ncol);

    for (i = 0; i < npar; ++i)
        p->s[i] = sqrt(2 * kBT * p->g[i] / dt);
}

void pair_get_view_dpd(const PairParams *p, PairDPD *v) {
    enum {PID=0};
    UC(check_dpd(p));
    v->a = p->a[PID];
    v->g = p->g[PID];
    v->s = p->s[PID];
    v->spow = p->spow;
}

void pair_get_view_dpd_color(const PairParams *p, PairDPDC *v) {
    size_t sz;
    int ncol, npar;
    UC(check_dpd(p));
    ncol = v->ncolors = p->ncolors;
    npar = get_npar(ncol);
    sz = npar * sizeof(float);
    memcpy(v->a, p->a, sz);
    memcpy(v->g, p->g, sz);
    memcpy(v->s, p->s, sz);
    v->spow = p->spow;
}

void pair_get_view_dpd_mirrored(const PairParams *p, PairDPDCM *v) {
    int i, j, nc;
    UC(check_dpd(p));
    nc = v->ncolors = p->ncolors;
    for (i = 0; i < nc; ++i) {
        j = (i+1) * (i+2) / 2 - 1; /* take all i-i pairs */
        v->a[i] = p->a[j];
        v->g[i] = p->g[j];
        v->s[i] = p->s[j];
    }
    v->spow = p->spow;
}

void pair_get_view_dpd_lj(const PairParams *p, PairDPDLJ *v) {
    enum {PID=0};
    UC(check_dpd(p));
    UC(check_lj(p));
    v->a = p->a[PID];
    v->g = p->g[PID];
    v->s = p->s[PID];
    v->ljs = p->ljs;
    v->lje = p->lje;
    v->spow = p->spow;
}

void pair_get_view_adhesion(const PairParams *p, PairAdhesion *v) {
    UC(check_adhesion(p));
    v->k1 = p->k1;
    v->k2 = p->k2;
}
