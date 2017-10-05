static void reg(int f, int x, int y,  /**/ int *hx, int *hy) { /* register an edge */
    int j = f*md;
    while (hx[j] != -1) j++;
    hx[j] = x; hy[j] = y;
}

static int nxt(int i, int x, int *hx, int *hy) { /* next */
    i *= md;
    while (hx[i] != x) i++;
    return hy[i];
}

static void gen_a12(int i0, int *hx, int *hy, /**/ int *a1, int *a2) {
    int lo = i0*md, hi = lo + md, mi = hx[lo];
    int i;
    for (i = lo + 1; (i < hi) && (hx[i] != -1); i++)
        if (hx[i] < mi) mi = hx[i]; /* minimum */

    int c = mi, c0;
    i = lo;
    do {
        c     = nxt(i0, c0 = c, hx, hy);
        a1[i] = c0;
        a2[i] = nxt(c, c0, hx, hy);
        i++;
    }  while (c != mi);
}

static void setup(const char *r_templ, int4 *faces, int4 *tri, int *adj0, int *adj1) {
    off::faces(r_templ, faces);

    cH2D(tri, faces, nt);

    int hx[nv*md], hy[nv*md], a1[nv*md], a2[nv*md];
    int i;
    for (i = 0; i < nv*md; i++) hx[i] = a1[i] = a2[i] = -1;

    int4 t;
    for (int ifa = 0; ifa < nt; ifa++) {
        t = faces[ifa];
        int f0 = t.x, f1 = t.y, f2 = t.z;
        reg(f0, f1, f2,   hx, hy); /* register an edge */
        reg(f1, f2, f0,   hx, hy);
        reg(f2, f0, f1,   hx, hy);
    }
    for (i = 0; i < nv; i++) gen_a12(i, hx, hy, /**/ a1, a2);

    cH2D(adj0, a1, nv*md);
    cH2D(adj1, a2, nv*md);
}


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

static void shift2local(const int orig[3], float A[16]) {
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

static int setup_hst(const char *r_templ, const char *r_state, int nv, Particle *pp) {
    /* fills `pp' with RBCs for this processor */

    float rr0[3*MAX_VERT_NUM]; /* rbc template */
    off::vert(r_templ, rr0);

    int c, nc = 0;
    int mi[3], L[3] = {XS, YS, ZS};
    for (c = 0; c < 3; ++c) mi[c] = (m::coords[c] + 0.5) * L[c];

    float A[4*4]; /* 4x4 affice transformation matrix */
    FILE *f = fopen(r_state, "r");
    if (f == NULL) ERR("Could not open <%s>\n", r_state);
        
    while ( read_A(f, /**/ A) ) {
        shift2local(mi, /**/ A);
        if ( inside_subdomain(L, A) )
            transform(rr0, nv, A, &pp[nv*(nc++)]);
    }
    fclose(f);
    MSG("read %d rbcs", nc);
    return nc;
}

static void setup_from_pos(const char *r_templ, const char *r_state, int nv, /**/ Particle *pp, int *nc, int *n, /* storage */ Particle *pp_hst) {
    /* fills `pp' with RBCs for this processor */
    *nc = setup_hst(r_templ, r_state, nv, pp_hst);
    if (*nc) cH2D(pp, pp_hst, nv * *nc);
    m::Barrier(m::cart);
    *n = *nc * nv;
}

void gen_quants(const char *r_templ, const char *r_state, Quants *q) {
    sub::setup(r_templ, /**/ q->tri_hst, q->tri, q->adj0, q->adj1);
    ic::setup_from_pos(r_templ, r_state, q->nv, /**/ q->pp, &q->nc, &q->n, /*w*/ q->pp_hst);
}
