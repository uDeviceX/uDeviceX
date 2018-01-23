static void gen0(const Coords *coords, float *D, /**/ Sdf *sdf) {
    int M[3] = {XWM, YWM, ZWM}; /* margin and texture */
    int T[3] = {XTE, YTE, ZTE};

    UC(array3d_copy(XTE, YTE, ZTE, D, /**/ sdf->arr));
    UC(tex3d_ini(&sdf->tex));
    UC(tex3d_copy(sdf->arr, /**/ sdf->tex));
    UC(sub2tex_ini(coords, T, M, /**/ sdf->t));
}

static void gen1(const Coords *coords, int N[3], float *D0, float *D1, /**/ Sdf *sdf) {
    Tform *t;
    int M[3] = {XWM, YWM, ZWM}; /* margin and texture */
    int T[3] = {XTE, YTE, ZTE};

    UC(tform_ini(&t));
    UC(tex2sdf_ini(coords, T, N, M, /**/ t));
    UC(field::sample(t, N, D0,   T, /**/ D1));
    UC(gen0(coords, D1, sdf));

    UC(tform_fin(t));
}

static void gen2(const Coords *coords, int N[3], float *D0, /**/ Sdf *sdf) {
    int sz;
    float *D1;
    sz = sizeof(D1[0])*XTE*YTE*ZTE;
    UC(emalloc(sz, (void**)&D1));
    UC(gen1(coords, N, D0, D1, /**/ sdf));
    UC(efree(D1));
}

static void gen3(const Coords *coords, MPI_Comm cart, int N[3], float ext[3], float *D, /**/ Sdf *sdf) {
    enum {X, Y, Z};
    float sc, G; /* domain size in x ([G]lobal) */
    G = m::dims[X] * XS;
    sc = G / ext[X];
    UC(field::scale(N, sc, /**/ D));
    if (field_dumps) UC(field::dump(coords, cart, N, D));
    UC(gen2(coords, N, D, /**/ sdf));
}

void sdf_gen(const Coords *coords, MPI_Comm cart, Sdf *sdf) {
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
