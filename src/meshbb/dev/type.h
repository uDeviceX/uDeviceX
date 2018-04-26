// typedef double   real_t;
// typedef double3 real3_t;

typedef float   real_t;
typedef float3 real3_t;

struct rPa {
    real3_t r, v;
};

_S_ real3_t make_real3(real_t a, real_t b, real_t c) {
    real3_t r;
    r.x = a;
    r.y = b;
    r.z = c;
    return r;
}

_I_ rPa fetch_Part(int i, const Particle *pp) {
    enum {X, Y, Z};
    const Particle *p = &pp[i];
    rPa rp = {
        .r = make_real3((real_t) p->r[X], (real_t) p->r[Y], (real_t) p->r[Z]),
        .v = make_real3((real_t) p->v[X], (real_t) p->v[Y], (real_t) p->v[Z])
    };
    return rp;
}

_I_ void write_Part(const rPa *rp, int i, Particle *pp) {
    Particle p = {
        .r = {(float) rp->r.x, (float) rp->r.y, (float) rp->r.z},
        .v = {(float) rp->v.x, (float) rp->v.y, (float) rp->v.z}
    };
    pp[i] = p;
}
