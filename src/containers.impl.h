namespace Cont {
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

int setup_hst(int nv, Particle *pp) {
    /* fills `pp' with RBCs for this processor */
    const char *r_templ = "rbc.off", *r_state = "rbcs-ic.txt";

    float rr0[3*MAX_VERT_NUM]; /* rbc template */
    off::f2vert(r_templ, rr0);

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

int setup(Particle* pp, int nv, /* storage */ Particle *pp_hst) {
    /* fills `pp' with RBCs for this processor */
    int nc = setup_hst(nv, pp_hst);
    if (nc) cH2D(pp, pp_hst, nv * nc);
    MPI_Barrier(m::cart);
    return nc;
}

template <bool hst, typename T>
int remove(T* data, const int npb, const int nb, const int *e, const int ne) {
    /* remove data with indexes in `e' */
    /* data: nb blocks of size npb     */
    const bool GO = false, STAY = true;
    int ie, i0, i1;
    std::vector<bool> m(nb, STAY);
    for (ie = 0; ie < ne; ie++) m[e[ie]] = GO;

    for (i0 = i1 = 0; i0 < nb; i0++)
    if (m[i0] == STAY)
    {
        if (hst) memcpy(data + npb*i1, data + npb*i0, sizeof(T)*npb);
        else cD2D(data + npb*i1, data + npb*i0, npb);
	i1++;
    }
    const int nstay = i1;
    return nstay;
}

void rbc_dump(int nc, Particle *p, int* triplets,
              int nv, int nt, int id) {
    const char *format4ply = "ply/rbcs-%05d.ply";
    char buf[200];
    sprintf(buf, format4ply, id);
    if (m::rank == 0) mkdir("ply", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    ply_dump(buf, triplets, nc, nt, p, nv);
}
}
