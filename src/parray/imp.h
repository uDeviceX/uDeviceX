struct PaArray {
    const float *pp;
};

struct PaCArray {
    const float *pp;
    const int *cc;   /* colors */
};

static void parray_push(const Particle *pp, PaArray *a) {
    a->pp = (const float*)pp;
};

static void parray_push(const Particle *pp, const int *cc, PaCArray *a) {
    a->pp = (const float*)pp;
    c->cc = cc;
}
