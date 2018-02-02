static void compute_edg(Adj *adj, const float *rr, /**/ float *o) {
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

static void compute_area(Adj *adj, const float *rr, /**/ float *o) {
    int i, i0, i1, i2, valid, n;
    const float *r0, *r1, *r2;
    float r01[3], r12[3], r20[3];
    float a, b, c, A; /* edges and area */
    AdjMap m;
    n = adj_get_max(adj);
    for (i = 0; i < n; i++) {
        valid = adj_get_map(i, adj, /**/ &m);
        if (!valid) continue;
        i0 = m.i0; i1 = m.i1; i2 = m.i2;
        r0 = &rr[3*i0]; r1 = &rr[3*i1]; r2 = &rr[3*i2];
        diff(r0, r1, /**/ r01);
        diff(r1, r2, /**/ r12);
        diff(r2, r0, /**/ r20);
        a = vabs(r01); b = vabs(r12); c = vabs(r20);
        A = area_kahan(a, b, c);
        o[i] = A;
    }
}

static void compute_total_volume(Adj *adj, const float *rr, /**/ float *pV) {
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

static void compute_total_area(Adj *adj, const float* area, /**/ float *pA) {
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

void rbc_shape_ini(Adj *adj, const float *rr, /**/ RbcShape **pq) {
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
void rbc_shape_total_volume(RbcShape *q, /**/ float *pe) { *pe = q->V; }
