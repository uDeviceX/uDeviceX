static void alloc(int n, Fo *f) { Dalloc(&f->f, n); }
static void dealloc(Fo *f) { Dfree(f->f); }

void ini0(const char* path, int n, float *h, /**/ float *d) {

}

void ini(const char* path, int n, /**/ Fo *f) {
    float *d, *h; /* device and host */
    alloc(n, f);
    d = f->f;
    h = (float*) malloc(3*n*sizeof(float));
    ini0(path, n, /*w*/ h, /**/ d);
    free(h);
}

void fin(Fo *f) { dealloc(f); }

void apply(int nm, const Particle*, const Fo*, /**/ Force*) {

}
