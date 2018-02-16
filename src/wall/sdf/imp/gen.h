static void gen0(const Coords *coords, Field *F, /**/ Sdf *sdf) {
    enum {X, Y, Z};
    int M[3] = {XWM, YWM, ZWM}; /* margin and texture */
    int N[3];
    float *D;
    field_size(F, /**/  N);
    field_data(F, /**/ &D);
    UC(array3d_copy(N[X], N[Y], N[Z], D, /**/ sdf->arr));
    UC(tex3d_ini(&sdf->tex));
    UC(tex3d_copy(sdf->arr, /**/ sdf->tex));
    UC(sub2tex_ini(coords, N, M, /**/ sdf->t));
}

static void gen1(const Coords *coords, const int T[3], Field *F0, /**/ Sdf *sdf) {
    Tform *t;
    int N[3];
    int M[3] = {XWM, YWM, ZWM}; /* margin and texture */
    Field *F1;

    UC(tform_ini(&t));
    field_size(F0, /**/ N);
    UC(tex2sdf_ini(coords, T, N, M, /**/ t));
    UC(field_sample(F0, t, T, /**/ &F1));
    UC(gen0(coords, F1, sdf));
    
    UC(field_fin(F1));
    UC(tform_fin(t));
}

static void gen2(const Coords *coords, Field *F, /**/ Sdf *sdf) {
    int3 Lte = sdf->Lte;
    int T[] = {Lte.x, Lte.y, Lte.z};
    UC(gen1(coords, T, F, /**/ sdf));
}

static void gen3(const Coords *coords, bool dump, MPI_Comm cart, Field *F, /**/ Sdf *sdf) {
    enum {X, Y, Z};
    float sc, G, ext[3];
    G = xdomain(coords);
    UC(field_extend(F, /**/ ext));
    sc = G / ext[X];
    UC(field_scale(F, sc));
    if (dump)
        UC(field_dump(F, coords, cart));
    UC(gen2(coords, F, /**/ sdf));
}

void sdf_gen(const Coords *coords, MPI_Comm cart, bool dump, Sdf *sdf) {
    enum {X, Y, Z};
    const char *f = "sdf.dat";
    Field *F;
    UC(field_ini(f, &F));
    UC(gen3(coords, dump, cart, F, /**/ sdf));
    UC(field_fin(F));
}
