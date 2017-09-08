void transform(float* rr0, int nv, float *A, /* output */ Particle *pp) {
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

int setup_hst(const char *r_templ, const char *r_state, int nv, Particle *pp) {
    /* fills `pp' with RBCs for this processor */

    float rr0[3*MAX_VERT_NUM]; /* rbc template */
    off::vert(r_templ, rr0);

    int c, nc = 0;
    int mi[3], L[3] = {XS, YS, ZS};
    for (c = 0; c < 3; ++c) mi[c] = (m::coords[c] + 0.5) * L[c];

    float A[4*4]; /* 4x4 affice transformation matrix */
    FILE *f = fopen(r_state, "r");
    if (f == NULL) {
        ERR("Could not open <%s>\n", r_state);
    }
        
    while ( read_A(f, /**/ A) ) {
        shift2local(mi, /**/ A);
        if ( inside_subdomain(L, A) )
            transform(rr0, nv, A, &pp[nv*(nc++)]);
    }
    fclose(f);
    MSG("read %d rbcs", nc);
    return nc;
}

void setup_from_pos(const char *r_templ, const char *r_state, int nv, /**/ Particle *pp, int *nc, int *n, /* storage */ Particle *pp_hst) {
    /* fills `pp' with RBCs for this processor */
    *nc = setup_hst(r_templ, r_state, nv, pp_hst);
    if (*nc) cH2D(pp, pp_hst, nv * *nc);
    m::Barrier(m::cart);
    *n = *nc * nv;
}
