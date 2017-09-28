typedef float   real_t;
typedef float3 real3_t;

struct rPa {
    real3_t r, v;
};

__device__ float3 make_real3(float a, float b, float c) {
    return make_float3(a, b, c);
}

__device__ double3 make_real3(double a, double b, double c) {
    return make_double3(a, b, c);
}

__device__ rPa P2rP(const Particle *p) {
    enum {X, Y, Z};
    rPa rp = {
        .r = make_real3((real_t) p->r[X], (real_t) p->r[Y], (real_t) p->r[Z]),
        .v = make_real3((real_t) p->v[X], (real_t) p->v[Y], (real_t) p->v[Z])
    };
    return rp;
}

__device__ Particle rP2P(const rPa *rp) {
    Particle p = {
        .r = {(float) rp->r.x, (float) rp->r.y, (float) rp->r.z},
        .v = {(float) rp->v.x, (float) rp->v.y, (float) rp->v.z}
    };
    return p;
}

/* use templates here because we might have mixed float/double
   see intersection routine                                    */

template <typename T1, typename T2, typename T3>
__device__ void diff(const T1 *a, const T2 *b, /**/ T3 *c) {
    c->x = a->x - b->x;
    c->y = a->y - b->y;
    c->z = a->z - b->z;
}

template <typename T1, typename T2>
__device__ double dot(const T1 *a, const T2 *b) {
    return a->x * b->x + a->y * b->y + a->z * b->z;
}

template <typename T1, typename T2, typename T3>
__device__ void cross(const T1 *a, const T2 *b, /**/ T3 *c) {
    c->x = a->y * b->z - a->z * b->y;
    c->y = a->z * b->x - a->x * b->z;
    c->z = a->x * b->y - a->y * b->x;
}

template <typename T1, typename T2>
__device__ void scalmult(const T1 *a, const real_t x, /**/ const T2 *b) {
    b->x = a->x * x;
    b->y = a->y * x;
    b->z = a->z * x;
}

template <typename T1, typename T2, typename T3, typename T4>
__device__ void apxb(const T1 *a, const T2 x, const T3 *b, /**/ T4 *c) {
    c->x = a->x + x * b->x;
    c->y = a->y + x * b->y;
    c->z = a->z + x * b->z;
}
