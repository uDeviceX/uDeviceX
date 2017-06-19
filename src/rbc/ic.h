void transform(float* rr0, int nv, float *A, /* output */ Particle* pp) {
    /* rr0: vertices of RBC template
       A: affice transfromation matrix */
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

int setup_hst(const char *r_templ, const char *r_state, int nv, Particle *pp) {
    /* fills `pp' with RBCs for this processor */

    float rr0[3*MAX_VERT_NUM]; /* rbc template */
    l::off::vert(r_templ, rr0);

    int i, j, c, nc = 0;
    int mi[3], L[3] = {XS, YS, ZS};
    for (c = 0; c < 3; ++c) mi[c] = (m::coords[c] + 0.5) * L[c];

    float A[4*4]; /* 4x4 affice transformation matrix */
    FILE *f = fopen(r_state, "r");
    if (f == NULL)
    {
        fprintf(stderr, "cont: Could not open <%s>\n", r_state);
        exit(1);
    }
        
    while (true) {
        for (i = 0; i < 4*4; i++) if (fscanf(f, "%f", &A[i]) != 1) goto done;
        for (c = 0; c < 3; c++) {
            j = 4 * c + 3;
            A[j] -= mi[c]; /* in local coordinates */
            if (2*A[j] < -L[c] || 2*A[j] > L[c]) goto next; /* not my RBC */
        }
        transform(rr0, nv, A, &pp[nv*(nc++)]);
    next: ;
    }
 done:
    fclose(f);
    MSG("read %d rbcs", nc);
    return nc;
}

int setup_from_states(const char *r_templ, const char *r_state, Particle* pp, int nv, /* storage */ Particle *pp_hst) {
    /* fills `pp' with RBCs for this processor */
    int nc = setup_hst(r_templ, r_state, nv, pp_hst);
    if (nc) cH2D(pp, pp_hst, nv * nc);
    l::m::Barrier(m::cart);
    return nc;
}
