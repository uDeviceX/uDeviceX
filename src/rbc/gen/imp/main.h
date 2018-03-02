static void mult(double A[16], const double a[4], /**/ double b[4]) {
    /* b = A x a */
    enum {X, Y, Z, W};    
    int c, i;
    for (i = c = 0; c < 4; c++) {
        b[c]  = A[i++]*a[X];
        b[c] += A[i++]*a[Y];
        b[c] += A[i++]*a[Z];
        b[c] += A[i++]*a[W];
    }
}

static void homogenius_multi(double A[16], const float a0[3], /**/ float b0[3]) {
    /* b = A x [a, 1] */
    enum {X, Y, Z, W};
    double a[4], b[4];
    a[X] = a0[X]; a[Y] = a0[Y]; a[Z] = a0[Z]; a[W] = 1;
    mult(A, a, /**/ b);
    b0[X] = b[X]; b0[Y] = b[Y]; b0[Z] = b[Z];
}

static void gen0(double A[16], const float *r, Particle *p) {
    enum {X, Y, Z};
    p->v[X] = p->v[Y] = p->v[Z] = 0;
    homogenius_multi(A, r, /**/ p->r);
}

static void gen1(double A[16], int nv, const float *rr, Particle *pp) {
    int i;
    for (i = 0; i < nv; i++) gen0(A, &rr[3*i], &pp[i]);
}
void rbc_gen0(int nv, const float *rr, const Matrices *matrices, /**/ int *pn, Particle *pp) {
    int i, n, nm;
    double *A;
    n = 0;
    nm = matrices_get_n(matrices);
    for (i = 0; i < nm; i++) {
        matrices_get(matrices, i, &A);
        gen1(A, nv, rr, &pp[n]); n += nv;
    }
    *pn = n;
}

static void shift(const Coords *c, Particle *p) {
    enum {X, Y, Z};
    float *r;
    r = p->r;
    r[X] = xg2xl(c, r[X]);
    r[Y] = yg2yl(c, r[Y]);
    r[Z] = zg2zl(c, r[Z]);
}
void rbc_shift(const Coords *c, int n, Particle *pp) {
    int i;
    for (i = 0; i < n; i++) shift(c, &pp[i]);
}

int rbc_gen(const Coords *coords, const float *rr0, const char *path, int nv, Particle *pp) {
    int n;
    Matrices *matrices;
    if (nv <= 0) ERR("nv <= 0");
    UC(matrices_read_filter(path, coords, /**/ &matrices));
    UC(matrices_log(matrices));
    UC(rbc_gen0(nv, rr0, matrices, /**/ &n, pp));
    UC(rbc_shift(coords, n, pp));
    UC(matrices_fin(matrices));
    return n / nv;
}
