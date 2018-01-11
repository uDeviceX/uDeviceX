static void alloc(int n, StretchForce *f) { Dalloc(&f->f, 3*n); }
static void dealloc(StretchForce *f) { Dfree(f->f); }

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
    msg_print("reading <%s>", path);
    FILE *f;
    UC(efopen(path, "r", /**/ &f));
    read(f, n , /**/ h);
    cH2D(d, h, 3*n);
    fclose(f);
}

static void ini1(const char* path, int n, /**/ StretchForce *f) {
    float *d, *h; /* device and host */
    alloc(n, f);
    d = f->f;
    UC(emalloc(3 * n * sizeof(float), (void**) &h));
    UC(ini0(path, n, /*w*/ h, /**/ d));
    free(h);
}

void rbc_stretch_ini(const char* path, int nv, /**/ StretchForce **fp) {
    StretchForce *f;
    UC(emalloc(sizeof(StretchForce), (void**) &f));
    UC(ini1(path, nv, f));
    f->nv = nv;
    *fp = f;
}

void rbc_stretch_fin(StretchForce *f) {
    dealloc(f);
    free(f);
}

void rbc_stretch_apply(int nm, const StretchForce *f, /**/ Force *ff) {
    int n, nv;
    nv = f->nv;
    n  = nm * nv;
    if (n) KL(dev::apply, (k_cnf(n)), (n, nv, f->f, /**/ ff));
}
