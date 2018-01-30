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

/* shift to local coordinates */
static void shift(const Coords *coords, /**/ float A[16]) {
    enum {X_, Y_, Z_};
    enum {
        X = 4*X_ + 3,
        Y = 4*Y_ + 3,
        Z = 4*Z_ + 3,
    };
    A[X] = xg2xl(coords, A[X]);
    A[Y] = yg2yl(coords, A[Y]);
    A[Z] = zg2zl(coords, A[Z]);
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

static int main0(const Coords *coords, float *rr0, const char *ic, int nv, Particle *pp) {
    int nc = 0;
    int L[3] = {XS, YS, ZS};

    float A[4*4]; /* 4x4 affice transformation matrix */
    FILE *f;
    UC(efopen(ic, "r", /**/ &f));

    while ( read_A(f, /**/ A) ) {
        shift(coords, /**/ A);
        if ( inside_subdomain(L, A) )
            transform(rr0, nv, A, &pp[nv*(nc++)]);
    }
    assert_nc(nc);
    UC(efclose(f));
    msg_print("read %d rbcs", nc);
    return nc;
}

static void vert(const char *f, int n0, /**/ const float *vert) {
    int n;
    UC(off_read_vert(f, n0, /**/ &n, vert));
    if (n0 != n)
        ERR("wrong vert number in <%s> : %d != %d", f, n0, n);
}

int rbc_gen(const Coords *coords, const float *vv, const char *ic, int nv, /**/ Particle *pp) {
    /* vv : vertices : x y z x y z, ... */
    float *rr0;
    int nc;
    UC(vert(cell, nv, /**/ rr0));
    nc = main0(coords, vv, ic, nv, pp);
    return nc;
}
