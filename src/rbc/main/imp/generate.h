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

static void assert_nc(int nc) {
    if (nc < MAX_CELL_NUM) return;
    ERR("nc = %d >= MAX_CELL_NUM = %d", nc, MAX_CELL_NUM);
}

static int setup_hst0(float *rr0, const char *r_state, int nv, Particle *pp) {
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
    assert_nc(nc);
    fclose(f);
    MSG("read %d rbcs", nc);
    return nc;
}

static void vert(const char *f, int n0, /**/ float *vert) {
    int n;
    n = off::vert(f, n0, vert);
    if (n0 != n)
        ERR("wrong vert number in <%s> : %d != %d", f, n0, n);
}
static int setup_hst(const char *r_templ, const char *r_state, int nv, /**/ Particle *pp) {
    float *rr0;
    int nc;
    rr0 = (float*) malloc(3*nv*sizeof(float));

    vert(r_templ, nv, /**/ rr0);
    nc = setup_hst0(rr0, r_state, nv, pp);

    free(rr0);
    return nc;
}

static void setup_from_pos(const char *r_templ, const char *r_state, int nv, /**/ Particle *pp, int *nc, int *n, /* storage */ Particle *pp_hst) {
    /* fills `pp' with RBCs for this processor */
    *nc = setup_hst(r_templ, r_state, nv, pp_hst);
    if (*nc) cH2D(pp, pp_hst, nv * *nc);
    MC(m::Barrier(m::cart));
    *n = *nc * nv;
}

static void gen_ids(long nc, /**/ int *ii) {
    long i, i0 = 0;
    MC(m::Exscan(&nc, &i0, 1, MPI_LONG, MPI_SUM, m::cart));
    for (i = 0; i < nc; ++i)
        ii[i] = i + i0;
}

void gen_quants(const char *r_templ, const char *r_state, /**/ Quants *q) {
    int md, nt, nv;
    md = RBCmd;
    nt = RBCnt;
    nv = RBCnv;
    setup(md, nt, nv, r_templ, /**/ q->tri_hst, q->tri, q->adj0, q->adj1);
    setup_from_pos(r_templ, r_state, q->nv, /**/ q->pp, &q->nc, &q->n, /*w*/ q->pp_hst);

    if (rbc_ids)
        gen_ids(q->nc, /**/ q->ii);
}
