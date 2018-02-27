static void diff(const float *a, const float *b, /**/ float *c) {
    enum {X, Y, Z};
    c[X] = a[X] - b[X];
    c[Y] = a[Y] - b[Y];
    c[Z] = a[Z] - b[Z];
}
static double vabs(float *a) {
    enum {X, Y, Z};
    double r;
    r = a[X]*a[X] + a[Y]*a[Y] + a[Z]*a[Z];
    return sqrt(r);
}
static void compute_edg(const Adj *adj, const float *rr, /**/ float *o) {
    int i, valid, n;
    const float *r0, *r1;
    float r01[3];
    float a;
    AdjMap m;
    n = adj_get_max(adj);
    for (i = 0; i < n; i++) {
        valid = adj_get_map(i, adj, /**/ &m);
        if (!valid) continue;
        r0 = &rr[3*m.i0]; r1 = &rr[3*m.i1];
        diff(r0, r1, /**/ r01);
        a = vabs(r01);
        o[i] = a;
    }
}

static void copy(const float f[3], double d[3]) {
    enum {X, Y, Z};
    d[X] = f[X]; d[Y] = f[Y]; d[Z] = f[Z];
}
static double area(const float af[3], const float bf[3], const float cf[3]) {
    double ad[3], bd[3], cd[3];
    copy(af, ad); copy(bf, bd); copy(cf, cd);
    return tri_hst::area_kahan(ad, bd, cd);
}
static void compute_area(const Adj *adj, const float *rr, /**/ float *o) {
    int i, i0, i1, i2, valid, n;
    const float *r0, *r1, *r2;
    AdjMap m;
    n = adj_get_max(adj);
    for (i = 0; i < n; i++) {
        valid = adj_get_map(i, adj, /**/ &m);
        if (!valid) continue;
        i0 = m.i0; i1 = m.i1; i2 = m.i2;
        r0 = &rr[3*i0]; r1 = &rr[3*i1]; r2 = &rr[3*i2];
        o[i] = area(r0, r1, r2);
    }
}

static void compute_total_volume(const Adj *adj, const float *rr, /**/ float *pV) {
    enum {X, Y, Z};
    int i, i0, i1, i2, valid, n;
    const float *r0, *r1, *r2;
    float x0, y0, z0, x1, y1, z1, x2, y2, z2;
    AdjMap m;
    double V;
    V = 0;
    n = adj_get_max(adj);
    for (i = 0; i < n; i++) {
        valid = adj_get_map(i, adj, /**/ &m);
        if (!valid) continue;
        i0 = m.i0; i1 = m.i1; i2 = m.i2;
        r0 = &rr[3*i0]; r1 = &rr[3*i1]; r2 = &rr[3*i2];
        x0 = r0[X]; y0 = r0[Y]; z0 = r0[Z];
        x1 = r1[X]; y1 = r1[Y]; z1 = r1[Z];
        x2 = r2[X]; y2 = r2[Y]; z2 = r2[Z];
        V += (x0*y1-x1*y0)*z2+x2*(y0*z1-y1*z0)+y2*(x1*z0-x0*z1);
    }
    V /= 6; /* from the formula */
    V /= 3; /* every triangle was visited three times */
    msg_print("V: %g", V);
    *pV = V;
}

static void compute_total_area(const Adj *adj, const float* area, /**/ float *pA) {
    int n, i, valid;
    AdjMap m;
    double A;

    A = 0;
    n = adj_get_max(adj);
    for (i = 0; i < n; i++) {
        valid = adj_get_map(i, adj, /**/ &m);
        if (!valid) continue;
        A += area[i];
    }
    A /= 3; /* every triangle was visited three times */
    *pA = A;
}

void rbc_shape_ini(const Adj *adj, const float *rr, /**/ RbcShape **pq) {
    int n;
    RbcShape *q;
    n = adj_get_max(adj);
    EMALLOC(1, &q);
    EMALLOC(n, &q->edg);
    EMALLOC(n, &q->area);

    compute_edg(adj, rr, /**/ q->edg);
    compute_area(adj, rr, /**/ q->area);
    compute_total_area(adj, q->area, /**/ &q->A);
    compute_total_volume(adj, rr, /**/ &q->V);
    
    msg_print("A: %g", q->A);

    *pq = q;
}

void rbc_shape_fin(RbcShape *q) {
    EFREE(q->edg);
    EFREE(q->area);
    EFREE(q);
}

void rbc_shape_edg  (RbcShape *q, /**/ float** pe) { *pe = q->edg; }
void rbc_shape_area (RbcShape *q, /**/ float** pe) { *pe = q->area; }

void rbc_shape_total_area(RbcShape *q, /**/ float *pe)   { *pe = q->A; }
void rbc_shape_total_volume(RbcShape *q, /**/ float *pe) {
    ERR("rbc_shape_total_volume is not implimented");
    *pe = q->V;
}
