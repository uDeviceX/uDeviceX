struct Tex { /* simplifies communication between gen[0123..] */
    cudaArray *a;
    tex3Dca   *t;
};

static void gen0(float *D, /**/ struct Tex te) {
    cudaMemcpy3DParms copyParams;
    memset(&copyParams, 0, sizeof(copyParams));
    copyParams.srcPtr = make_cudaPitchedPtr((void*)D, XTE * sizeof(float), XTE, YTE);
    copyParams.dstArray = te.a;
    copyParams.extent = make_cudaExtent(XTE, YTE, ZTE);
    copyParams.kind = H2D;
    CC(cudaMemcpy3D(&copyParams));
    te.t->setup(te.a);
}

static void gen1(int N[3], float *D0, float *D1, /**/ struct Tex te) {
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
    UC(gen0(D1, te));
}

static void gen2(int N[3], float* D0, /**/ struct Tex te) {
    int sz;
    float *D1;
    sz = sizeof(float)*XTE*YTE*ZTE;
    UC(emalloc(sz, (void**)&D1));
    UC(gen1(N, D0, D1, /**/ te));
    UC(efree(D1));
}

static void gen3(MPI_Comm cart, int N[3], float ext[3], float* D, /**/ struct Tex te) {
    enum {X, Y, Z};
    float sc, G; /* domain size in x ([G]lobal) */
    G = m::dims[X] * XS;
    sc = G / ext[X];
    UC(field::scale(N, sc, /**/ D));
    if (field_dumps) UC(field::dump(cart, N, D));
    UC(gen2(N, D, /**/ te));
}

static void gen(MPI_Comm cart, cudaArray *arrsdf, tex3Dca *texsdf) {
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
    UC(gen3(cart, N, ext, D, /**/ te));
    delete[] D;
}

void gen(MPI_Comm cart, Sdf *sdf) {
    gen(cart, sdf->arrsdf, &sdf->texsdf);
}
