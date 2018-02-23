static void transform(const float *rr0, int nv, float *A, /**/ Particle *pp) {
    int iv, c, i;
    float *r, *v;
    const float *r0;
    /* rr0: vertices of RBC template; A: affine transformation matrix */
    for (iv = 0; iv < nv; iv++) {
        r = pp[iv].r;
        v = pp[iv].v;
        r0 = &rr0[3*iv];
        for (c = 0, i = 0; c < 3; c++) {
            r[c] += A[i++]*r0[0]; /* matrix transformation */
            r[c] += A[i++]*r0[1];
            r[c] += A[i++]*r0[2];
            r[c] += A[i++];

            v[c] = 0;
        }
    }
}

static bool read_A(FILE *f, float A[16]) {
    int i;
    for (i = 0; i < 4*4; i++)
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

int rbc_gen(const Coords *coords, const float *rr0, const char *ic, int nv, Particle *pp) {
    int nc = 0;
    int L[3] = {xs(coords), ys(coords), zs(coords)};
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
