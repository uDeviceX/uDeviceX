static void transform(float* rr0, int nv, float *A, /* output */ Particle *pp) {
    /* rr0: vertices of RBC template; A: affine transformation matrix */
    for (int iv = 0; iv < nv; iv++) {
        float  *r = pp[iv].r, *v = pp[iv].v;
        float *r0 = &rr0[3*iv];
        for (int c = 0, i = 0; c < 3; c++) {
            r[c] += A[i++]*r0[0]; /* matrix transformation */
            r[c] += A[i++]*r0[1];
            r[c] += A[i++]*r0[2];
            r[c] += A[i++];

            v[c] = 0;
        }
    }
}

static bool read_A(FILE *f, float A[16]) {
    for (int i = 0; i < 4*4; i++)
        if (fscanf(f, "%f", &A[i]) != 1) return false;
    return true;
}

static void shift(const int orig[3], /**/ float A[16]) {
    int c, j;
    for (c = 0; c < 3; c++) {
        j = 4 * c + 3;
        A[j] -= orig[c]; /* in local coordinates */
    }
}

static bool inside_subdomain(const int L[3], const float A[16]) {
    int c, j;
    for (c = 0; c < 3; c++) {
        j = 4 * c + 3;
        if (2*A[j] < -L[c] || 2*A[j] >= L[c]) return false; /* not my RBC */
    }
    return true;
}

static void assert_nc(int nc) {
    if (nc < MAX_CELL_NUM) return;
    ERR("nc = %d >= MAX_CELL_NUM = %d", nc, MAX_CELL_NUM);
}

static int main0(float *rr0, const char *ic, int nv, Particle *pp) {
    int c, nc = 0;
    int mi[3], L[3] = {XS, YS, ZS};
    for (c = 0; c < 3; ++c) mi[c] = (m::coords[c] + 0.5) * L[c];

    float A[4*4]; /* 4x4 affice transformation matrix */
    FILE *f;
    UC(efopen(ic, "r", /**/ &f));

    while ( read_A(f, /**/ A) ) {
        shift(mi, /**/ A);
        if ( inside_subdomain(L, A) )
            transform(rr0, nv, A, &pp[nv*(nc++)]);
    }
    assert_nc(nc);
    UC(efclose(f));
    MSG("read %d rbcs", nc);
    return nc;
}

static void vert(const char *f, int n0, /**/ float *vert) {
    int n;
    UC(off::vert(f, n0, /**/ &n, vert));
    if (n0 != n)
        ERR("wrong vert number in <%s> : %d != %d", f, n0, n);
}

int main(const char *cell, const char *ic, int nv, /**/ Particle *pp) {
    float *rr0;
    int nc;
    UC(emalloc(3*nv*sizeof(float), (void**) &rr0));

    UC(vert(cell, nv, /**/ rr0));
    nc = main0(rr0, ic, nv, pp);

    free(rr0);
    return nc;
}
