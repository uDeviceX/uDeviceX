#include "dbg/error.h"
#include "dbg/dev.h"
namespace dev {
static __device__ void valid_f(float fx, float fy, float fz) {
    enum {X, Y, Z};
    bool verbose = true;
    Force f;
    f.f[X] = fx; f.f[Y] = fy; f.f[Z] = fz;
    assert(dbg::dev::valid_f(&f, verbose) == dbg::err::NONE);
}

static __device__ void f3xyz(float3 r, float *x, float *y, float *z) {
    *x = r.x; *y = r.y; *z = r.z;
}

static __device__ bool valid_r(float3 r) {
    float x, y, z;
    bool verbose = true;
    f3xyz(r,  &x,  &y,  &z);
    return dbg::dev::valid_unpacked_pos_pu(x, y, z, verbose) == dbg::err::NONE;
}

static __device__ bool valid_v(float3 v) {
    float x, y, z;
    bool verbose = true;
    f3xyz(v,  &x,  &y,  &z);
    return dbg::dev::valid_vel3(x, y, z, verbose);
}

#define msg(fmt, ...) printf("%s:%d: " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__)
static __device__ void check0(bool r1, bool r2, bool v1, bool v2) {
    msg("pos: loc, rem: %d %d", r1, r2);
    msg("vel: loc, rem: %d %d", v1, v2);
}
#undef msg

static __device__ void check1(bool r1, bool r2, bool v1, bool v2) { /* statuses */
    if (r1 && r2 && v1 && v2) return;
    check0(r1, r2, v1, v2);
    assert(0);
}

static __device__ void check(float3 r1, float3 r2, float3 v1, float3 v2) {
    check1(valid_r(r1), valid_r(r2), valid_v(v1), valid_v(v2));
}

static __device__ void pair0(const Pa l, const Pa r, float rnd, /**/ float *fx, float *fy, float *fz) {
    /* pair force ; l, r: local and remote */
    float3 r1, r2, v1, v2, f;
    r1 = make_float3( l.x,  l.y,  l.z); r2 = make_float3( r.x,  r.y,  r.z);
    v1 = make_float3(l.vx, l.vy, l.vz); v2 = make_float3(r.vx, r.vy, r.vz);

    check(r1, r2, v1, v2);
    
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
