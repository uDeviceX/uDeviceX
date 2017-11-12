static void   alloc(int n, Fo *f) { Dalloc(&f->f, 3*n); }
static void   dealloc(Fo *f) { Dfree(f->f); }
static float *halloc(int n)  { /* host alloc */
    return (float*)emalloc(3*n*sizeof(float));
}

static int read3(FILE *f, float *h) {
    enum {X, Y, Z};
    int n;
    n = fscanf(f, "%f %f %f\n", &h[X], &h[Y], &h[Z]);
    return n;
}
static void read(FILE *f, int n, /**/ float *h) {
    int i;
    i = 0;
    for (/**/ ; i < n && read3(f, h) == 3;   i++, h += 3)
        ;
    if (i != n) ERR("got %d != %d lines", i, n);
}
static void ini0(const char* path, int n, float *h, /**/ float *d) {
    MSG("reading <%s>", path);
    FILE *f = efopen(path, "r");
    read(f, n , /**/ h);
    cH2D(d, h, n);
    fclose(f);
}
void ini1(const char* path, int n, /**/ Fo *f) {
    float *d, *h; /* device and host */
    alloc(n, f);
    d = f->f;
    h = halloc(n);
    ini0(path, n, /*w*/ h, /**/ d);
    free(h);
}
void ini(const char* path, int n, /**/ Fo **fp) {
    Fo *f;
    f = (Fo*) malloc(sizeof(Fo));
    ini1(path, n, f);
    *fp = f;
}

void fin(Fo *f) {
    dealloc(f);
    free(f);
}

void apply(int nm, const Particle*, const Fo*, /**/ Force*) {

}
