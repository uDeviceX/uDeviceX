struct Tex { /* simplifies communication between ini[0123..] */
    cudaArray *a;
    tex3Dca   *t;
};

static void ini0(float *D, /**/ struct Tex te) {
    cudaMemcpy3DParms copyParams;
    memset(&copyParams, 0, sizeof(copyParams));
    copyParams.srcPtr = make_cudaPitchedPtr((void*)D, XTE * sizeof(float), XTE, YTE);
    copyParams.dstArray = te.a;
    copyParams.extent = make_cudaExtent(XTE, YTE, ZTE);
    copyParams.kind = H2D;
    CC(cudaMemcpy3D(&copyParams));
    te.t->setup(te.a);
}

static void ini1(int N[3], float *D0, float *D1, /**/ struct Tex te) {
    int c;
    int L[3] = {XS, YS, ZS};
    int M[3] = {XWM, YWM, ZWM}; /* margin and texture */
    int T[3] = {XTE, YTE, ZTE};
    float G; /* domain size ([g]lobal) */
    float lo; /* left edge of subdomain */
    float org[3], spa[3]; /* origin and spacing */
    for (c = 0; c < 3; ++c) {
        G = m::dims[c] * L[c];
        lo = m::coords[c] * L[c];
        spa[c] = N[c] * (L[c] + 2 * M[c]) / G / T[c];
        org[c] = N[c] * (lo - M[c]) / G;
    }
    UC(field::sample(org, spa, N, D0,   T, /**/ D1));
    UC(ini0(D1, te));
}

static void ini2(int N[3], float* D0, /**/ struct Tex te) {
    int sz;
    float *D1;
    sz = sizeof(float)*XTE*YTE*ZTE;
    UC(emalloc(sz, (void**)&D1));
    UC(ini1(N, D0, D1, /**/ te));
    UC(efree(D1));
}

static void ini3(MPI_Comm cart, int N[3], float ext[3], float* D, /**/ struct Tex te) {
    enum {X, Y, Z};
    float sc, G; /* domain size in x ([G]lobal) */
    G = m::dims[X] * XS;
    sc = G / ext[X];
    UC(field::scale(N, sc, /**/ D));
    if (field_dumps) UC(field::dump(cart, N, D));
    UC(ini2(N, D, /**/ te));
}

void ini(MPI_Comm cart, cudaArray *arrsdf, tex3Dca *texsdf) {
    enum {X, Y, Z};
    float *D;     /* data */
    int N[3];     /* size of D */
    float ext[3]; /* extent */
    int n;
    char f[] = "sdf.dat";
    struct Tex te {arrsdf, texsdf};

    UC(field::ini_dims(f, /**/ N, ext));
    n = N[X] * N[Y] * N[Z];
    D = new float[n];
    UC(field::ini_data(f, n, /**/ D));
    UC(ini3(cart, N, ext, D, /**/ te));
    delete[] D;
}

/* sort solvent particle into remaining in solvent and turning into wall according to keys (all on hst) */
static void split_wall_solvent(const int *keys, /*io*/ int *s_n, Particle *s_pp, /**/ int *w_n, Particle *w_pp) {
    int n = *s_n;
    Particle p;
    int k, ia = 0, is = 0, iw = 0; /* all, solvent, wall */

    for (ia = 0; ia < n; ++ia) {
        k = keys[ia];
        p = s_pp[ia];
        
        if      (k == W_BULK) s_pp[is++] = p;
        else if (k == W_WALL) w_pp[iw++] = p;
    }
    *s_n = is;
    *w_n = iw;
}
/* sort solvent particle (dev) into remaining in solvent (dev) and turning into wall (hst)*/
void bulk_wall(const tex3Dca texsdf, /*io*/ Particle *s_pp, int* s_n,
               /*o*/ Particle *w_pp, int *w_n) {
    int n = *s_n, *labels;
    Particle *s_pp_hst;
    UC(emalloc(n*sizeof(Particle), (void**) &s_pp_hst));
    UC(emalloc(n*sizeof(int),      (void**)&labels));
    
    UC(label::hst(texsdf, n, s_pp, labels));
    cD2H(s_pp_hst, s_pp, n);

    UC(split_wall_solvent(labels, /*io*/ s_n, s_pp_hst, /**/ w_n, w_pp));
    cH2D(s_pp, s_pp_hst, *s_n);
                       
    UC(efree(s_pp_hst));
    UC(efree(labels));
}

/* bulk predicate : is in bulk? */
static bool bulkp(int *keys, int i) { return keys[i] == W_BULK; }
static int who_stays0(int *keys, int nc, int nv, /*o*/ int *stay) {
    int c, v;  /* cell and vertex */
    int s = 0; /* how many stays? */
    for (c = 0; c < nc; ++c) {
        v = 0;
        while (v  < nv && bulkp(keys, v + nv * c)) v++;
        if    (v == nv) stay[s++] = c;
    }
    return s;
}
int who_stays(const tex3Dca texsdf, Particle *pp, int n, int nc, int nv, /**/ int *stay) {
    int *labels;
    UC(emalloc(n*sizeof(int), (void**)&labels));
    UC(label::hst(texsdf, n, pp, /**/ labels));
    nc = who_stays0(labels, nc, nv, /**/ stay);
    efree(labels);
    return nc;
}
