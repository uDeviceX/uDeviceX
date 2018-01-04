static void gen0(float *D, /**/ Sdf *sdf) {
    UC(array3d_copy(XTE, YTE, ZTE, D, /**/ sdf->arr));

    UC(tex3d_ini(&sdf->tex));
    UC(tex3d_copy(sdf->arr, /**/ sdf->tex));
}

static void gen1(int N[3], float *D0, float *D1, /**/ Sdf *sdf) {
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
    UC(gen0(D1, sdf));
}

static void gen2(int N[3], float *D0, /**/ Sdf *sdf) {
    int sz;
    float *D1;
    sz = sizeof(D1[0])*XTE*YTE*ZTE;
    UC(emalloc(sz, (void**)&D1));
    UC(gen1(N, D0, D1, /**/ sdf));
    UC(efree(D1));
}

static void gen3(Coords *coords, MPI_Comm cart, int N[3], float ext[3], float *D, /**/ Sdf *sdf) {
    enum {X, Y, Z};
    float sc, G; /* domain size in x ([G]lobal) */
    G = m::dims[X] * XS;
    sc = G / ext[X];
    UC(field::scale(N, sc, /**/ D));
    if (field_dumps) UC(field::dump(*coords, cart, N, D));
    UC(gen2(N, D, /**/ sdf));
}

void gen(Coords *coords, MPI_Comm cart, Sdf *sdf) {
    enum {X, Y, Z};
    float *D;     /* data */
    int N[3];     /* size of D */
    float ext[3]; /* extent */
    int n;
    const char *f = "sdf.dat";

    UC(field::ini_dims(f, /**/ N, ext));
    n = N[X] * N[Y] * N[Z];
    UC(emalloc(n*sizeof(D[0]), (void**)&D));
    UC(field::ini_data(f, n, /**/ D));
    UC(gen3(coords, cart, N, ext, D, /**/ sdf));
    UC(efree(D));
}
