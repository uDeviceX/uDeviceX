// typedef double   real_t;
// typedef double3 real3_t;

typedef float   real_t;
typedef float3 real3_t;

struct rPa {
    real3_t r, v;
};

static __device__ float3 make_real3(float a, float b, float c) {
    return make_float3(a, b, c);
}

static __device__ double3 make_real3(double a, double b, double c) {
    return make_double3(a, b, c);
}

static __device__ rPa P2rP(const Particle *p) {
    enum {X, Y, Z};
    rPa rp = {
        .r = make_real3((real_t) p->r[X], (real_t) p->r[Y], (real_t) p->r[Z]),
        .v = make_real3((real_t) p->v[X], (real_t) p->v[Y], (real_t) p->v[Z])
    };
    return rp;
}

static __device__ Particle rP2P(const rPa *rp) {
    Particle p = {
        .r = {(float) rp->r.x, (float) rp->r.y, (float) rp->r.z},
        .v = {(float) rp->v.x, (float) rp->v.y, (float) rp->v.z}
    };
    return p;
}
