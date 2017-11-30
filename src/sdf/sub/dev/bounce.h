static __device__ float fst(float2 *t) { return t->x; }
static __device__ float scn(float2 *t) { return t->y; }
static __device__ void p2rv(float2 *p, int i, /**/ float r[3], float v[3]) {
    enum {X, Y, Z};
    p += 3 * i;
    r[X] = fst(p);   r[Y] = scn(p++); r[Z] = fst(p);
    v[X] = scn(p++); v[Y] = fst(p);   v[Z] = scn(p);
}
static __device__ void rv2p(float r[3], float v[3], int i, /**/ float2 *p) {
    enum {X, Y, Z};
    p += 3 * i;
    p->x   = r[X]; p++->y = r[Y]; p->x = r[Z];
    p++->y = v[X]; p->x   = v[Y]; p->y = v[Z];
}

static __device__ float3 ugrad_sdf(const tex3Dca texsdf, float x, float y, float z) {
    int L[3] = {XS, YS, ZS};
    int M[3] = {XWM, YWM, ZWM};
    int T[3] = {XTE, YTE, ZTE};
    int tc[3];
    float fcts[3], r[3] = {x, y, z};
    for (int c = 0; c < 3; ++c)
        tc[c] = T[c] * (r[c] + L[c] / 2 + M[c]) / (L[c] + 2 * M[c]);
    for (int c = 0; c < 3; ++c)
        fcts[c] = T[c] / (2 * M[c] + L[c]);

#define tex0(ix, iy, iz) (texsdf.fetch(tc[0] + ix, tc[1] + iy, tc[2] + iz))
    float myval = tex0(0, 0, 0);
    float gx = fcts[0] * (tex0(1, 0, 0) - myval);
    float gy = fcts[1] * (tex0(0, 1, 0) - myval);
    float gz = fcts[2] * (tex0(0, 0, 1) - myval);
#undef tex0

    return make_float3(gx, gy, gz);
}

static __device__ float3 grad_sdf(const tex3Dca texsdf, float x, float y, float z) {
    float gx, gy, gz;
    int L[3] = {XS, YS, ZS};
    int M[3] = {XWM, YWM, ZWM};
    int T[3] = {XTE, YTE, ZTE};
    float tc[3], r[3] = {x, y, z};
    for (int c = 0; c < 3; ++c)
        tc[c] = T[c] * (r[c] + L[c] / 2 + M[c]) / (L[c] + 2 * M[c]) - 0.5;

#define tex0(ix, iy, iz) (texsdf.fetch(tc[0] + ix, tc[1] + iy, tc[2] + iz))
    gx = tex0(1, 0, 0) - tex0(-1,  0,  0);
    gy = tex0(0, 1, 0) - tex0( 0, -1,  0);
    gz = tex0(0, 0, 1) - tex0( 0,  0, -1);
#undef tex0
    float ggmag = sqrt(gx*gx + gy*gy + gz*gz);
    if (ggmag > 1e-6) { gx /= ggmag; gy /= ggmag; gz /= ggmag; }
    return make_float3(gx, gy, gz);
}

static __device__ void bounce1(const tex3Dca texsdf, float currsdf,
                               float &x, float &y, float &z,
                               float &vx, float &vy, float &vz) {
    float x0 = x - vx*dt, y0 = y - vy*dt, z0 = z - vz*dt;
    if (sdf(texsdf, x0, y0, z0) >= 0) {
        float3 dsdf = grad_sdf(texsdf, x, y, z); float sdf0 = currsdf;
        x -= sdf0 * dsdf.x; y -= sdf0 * dsdf.y; z -= sdf0 * dsdf.z;
        for (int l = 8; l >= 1; --l) {
            if (sdf(texsdf, x, y, z) < 0) {
                k_wvel::bounce_vel(x, y, z, &vx, &vy, &vz); return;
            }
            float jump = 1.1f * sdf0 / (1 << l);
            x -= jump * dsdf.x; y -= jump * dsdf.y; z -= jump * dsdf.z;
        }
    }

#define rr(t) make_float3(x + vx*t, y + vy*t, z + vz*t)
#define small(phi) (fabs(phi) < 1e-6)
    float3 r, dsdf; float phi, dphi, t = 0;
    r = rr(t); phi = currsdf;
    dsdf = ugrad_sdf(texsdf, r.x, r.y, r.z);
    dphi = vx*dsdf.x + vy*dsdf.y + vz*dsdf.z; if (small(dphi)) goto giveup;
    t -= phi/dphi;                            if (t < -dt) t = -dt; if (t > 0) t = 0;

    r = rr(t); phi = sdf(texsdf, r.x, r.y, r.z);
    dsdf = ugrad_sdf(texsdf, r.x, r.y, r.z);
    dphi = vx*dsdf.x + vy*dsdf.y + vz*dsdf.z; if (small(dphi)) goto giveup;
    t -= phi/dphi;                            if (t < -dt) t = -dt; if (t > 0) t = 0;
#undef rr
#undef small
 giveup:
    float xw = x + t*vx, yw = y + t*vy, zw = z + t*vz;
    x += 2*t*vx; y += 2*t*vy; z += 2*t*vz;
    k_wvel::bounce_vel(xw, yw, zw, &vx, &vy, &vz);
    if (sdf(texsdf, x, y, z) >= 0) {x = x0; y = y0; z = z0;}
}

__global__ void bounce(const tex3Dca texsdf, int n, /**/ float2 *const pp) {
    enum {X, Y, Z};
    float s, currsdf, r[3], v[3];
    int i;
    i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= n) return;
    p2rv(pp, i, /**/ r, v);
    s = cheap_sdf(texsdf, r[X], r[Y], r[Z]);
    if (s >= -1.7320 * XSIZE_WALLCELLS / XTE) {
        currsdf = sdf(texsdf, r[X], r[Y], r[Z]);
        if (currsdf >= 0) {
            bounce1(texsdf, currsdf, /*io*/ r[X], r[Y], r[Z], v[X], v[Y], v[Z]);
            rv2p(r, v, i, /**/ pp);
        }
    }
}
