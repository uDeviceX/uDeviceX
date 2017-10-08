static __device__ void gen0(Pa A, Pa B, float ma, float mb, float rnd, /**/ Fo *f) {
    float s; /* scale */
    s = ma * mb;
    force(A, B, rnd, /**/ f);
    f->x *= s; f->y *= s; f->z *= s;
}

static __device__ void gen(Pa A, Pa B, float rnd, /**/ Fo *f) {
    enum {O = SOLVENT_KIND, S = SOLID_KIND, W = WALL_KIND};
    int ka, kb;
    float m[3];
    float ma, mb;

    m[0] = 1; m[S] = rbc_mass; m[W] = 1;
    ka = A.kind; kb = B.kind;
    assert(ka < 3);
    assert(kb < 3);

    ma = m[ka]; mb = m[kb];

    gen0(A, B, ma, mb, rnd, /**/ f);
}
