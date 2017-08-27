#include "dbg/error.h"
#include "dbg/dev.h"
namespace dev {
static __device__ void valid_f(float fx, float fy, float fz) {
    enum {X, Y, Z};
    bool verbose = true;
    Force f;
    f.f[X] = fx; f.f[Y] = fy; f.f[Z] = fz;
    assert(dbg::dev::valid_f(&f, verbose));
}

static __device__ void pair0(const Pa l, const Pa r, float rnd, /**/ float *fx, float *fy, float *fz) {
    /* pair force ; l, r: local and remote */
    float3 r1, r2, v1, v2, f;
    r1 = make_float3( l.x,  l.y,  l.z); r2 = make_float3( r.x,  r.y,  r.z);
    v1 = make_float3(l.vx, l.vy, l.vz); v2 = make_float3(r.vx, r.vy, r.vz);
    f = forces::dpd(SOLID_TYPE, SOLVENT_TYPE, r1, r2, v1, v2, rnd); /* TODO: type */
    *fx = f.x; *fy = f.y; *fz = f.z;
    valid_f(*fx, *fy, *fz);
}

static __device__ void pair(const Pa l, const Pa r, float rnd, /**/
                            float *fx, float *fy, float *fz,
                            Fo f) {
    /* f[xyz]: local force; Fo f: remote force */
    float x, y, z; /* pair force */
    pair0(l, r, rnd, /**/ &x, &y, &z);
    *fx += x; *fy += y; *fz += z;
    atomicAdd(f.x, -x); atomicAdd(f.y, -y); atomicAdd(f.z, -z);
}
}
