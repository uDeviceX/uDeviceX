static void skip_line(FILE *f) {
    char l[BUFSIZ];
    UC(efgets(l, sizeof(l), f));
}

void sdf_field_ini_dims(const char *path, /**/ int N[3], float ext[3]) {
    FILE *f;
    char l[BUFSIZ];
    UC(efopen(path, "r", /**/ &f));
    UC(efgets(l, sizeof(l), f));
    sscanf(l, "%f %f %f", &ext[0], &ext[1], &ext[2]);
    UC(efgets(l, sizeof(l), f));
    sscanf(l, "%d %d %d", &N[0], &N[1], &N[2]);
    UC(efclose(f));
}
  
void sdf_field_ini_data(const char *path, int n, /**/ float *D) { /* read sdf file */
    FILE *f;
    UC(efopen(path, "r", /**/ &f));
    skip_line(f); skip_line(f);
    UC(efread(D, sizeof(D[0]), n, f));
    UC(efclose(f));
}

void sdf_field_scale(const int N[3], float s, /**/ float *D) {
    enum {X, Y, Z};
    int i, n;
    n = N[X]*N[Y]*N[Z];
    for (i = 0; i < n; i++) D[i] *= s;
}

static void dump0(const Coords *coords, const int N0[3], const float *D0, /**/ float *D1) {
    int L[3] = {xs(coords), ys(coords), zs(coords)};
    Tform *t;
    UC(tform_ini(&t));
    UC(out2sdf_ini(coords, N0, t));
    UC(sdf_field_sample(t, N0, D0,   L, /**/ D1));
    UC(tform_fin(t));
}

static void dump1(const Coords *coords, MPI_Comm cart, const int N[3], const float *D, /*w*/ float *W) {
    UC(dump0(coords, N, D, /**/ W));
    UC(io::field::scalar(coords, cart, W, "wall"));
}

void sdf_field_dump(const Coords *coords, MPI_Comm cart, const int N[], const float *D) {
    float *W;
    int ngrid = xs(coords) * ys(coords) * zs(coords);
    UC(emalloc(ngrid*sizeof(float), (void**) &W));
    UC(dump1(coords, cart, N, D, /*w*/ W));
    UC(efree(W));
}


/* new */
void field_size(const Field *q, /**/ int N[3]) {
    enum {X, Y, Z};
    N[X] = q->N[X]; N[Y] = q->N[Y]; N[Z] = q->N[Z];
}
void field_extension(const Field *q, /**/ float ext[3]) {
    enum {X, Y, Z};
    ext[X] = q->ext[X]; ext[Y] = q->ext[Y]; ext[Z] = q->ext[Z];
}
void field_data(const Field *q, /**/ float **pD) {
    *pD = q->D;
}
void field_scale(Field *q, float scale) {
    enum {X, Y, Z};
    float *D;
    int i, n, *N;
    N = q->N; D = q->D;
    n = N[X]*N[Y]*N[Z];
    for (i = 0; i < n; i++) D[i] *= scale;
}
