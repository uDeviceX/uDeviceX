static void gen0(const Coords *coords, const int T[3], float *D, /**/ Sdf *sdf) {
    int M[3] = {XWM, YWM, ZWM}; /* margin and texture */

    UC(array3d_copy(T[0], T[1], T[2], D, /**/ sdf->arr));
    UC(tex3d_ini(&sdf->tex));
    UC(tex3d_copy(sdf->arr, /**/ sdf->tex));
    UC(sub2tex_ini(coords, T, M, /**/ sdf->t));
}

static void gen1(const Coords *coords, const int T[3], int N[3], float *D0, float *D1, /**/ Sdf *sdf) {
    Tform *t;
    int M[3] = {XWM, YWM, ZWM}; /* margin and texture */

    UC(tform_ini(&t));
    UC(tex2sdf_ini(coords, T, N, M, /**/ t));
    UC(sdf_field_sample(t, N, D0,   T, /**/ D1));
    UC(gen0(coords, T, D1, sdf));

    UC(tform_fin(t));
}

static void gen2(const Coords *coords, int N[3], float *D0, /**/ Sdf *sdf) {
    int sz;
    float *D1;
    int3 Lte = sdf->Lte;
    int T[] = {Lte.x, Lte.y, Lte.z};
    sz = sizeof(D1[0]) * Lte.x * Lte.y * Lte.z;
    UC(emalloc(sz, (void**)&D1));
UC(gen1(coords, T, N, D0, D1, /**/ sdf));
    UC(efree(D1));
}

static void gen3(const Coords *coords, bool dump, MPI_Comm cart, int N[3], float ext[3], float *D, /**/ Sdf *sdf) {
    enum {X, Y, Z};
    float sc, G; /* domain size in x ([G]lobal) */
    G = xdomain(coords);
    sc = G / ext[X];
    UC(sdf_field_scale(N, sc, /**/ D));
    if (dump)
        UC(sdf_field_dump(coords, cart, N, D));
    UC(gen2(coords, N, D, /**/ sdf));
}

void sdf_gen(const Coords *coords, MPI_Comm cart, bool dump, Sdf *sdf) {
    enum {X, Y, Z};
    float *D;     /* data */
    int N[3];     /* size of D */
    float ext[3]; /* extent */
    int n;
    const char *f = "sdf.dat";

    UC(sdf_field_ini_dims(f, /**/ N, ext));
    n = N[X] * N[Y] * N[Z];
    UC(emalloc(n*sizeof(D[0]), (void**)&D));
    UC(sdf_field_ini_data(f, n, /**/ D));
    UC(gen3(coords, dump, cart, N, ext, D, /**/ sdf));
    UC(efree(D));
}
