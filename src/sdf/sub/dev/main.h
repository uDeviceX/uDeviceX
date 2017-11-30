namespace sdf { namespace sub { namespace dev {
__device__ float fst(float2 *t) { return t->x; }
__device__ float scn(float2 *t) { return t->y; }
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
static __device__ float sdf(const tex3Dca texsdf, float x, float y, float z) {
    int c;
    float t;
    float s000, s001, s010, s100, s101, s011, s110, s111;
    float s00x, s01x, s10x, s11x;
    float s0yx, s1yx;
    float szyx;

    float tc[3], lmbd[3], r[3] = {x, y, z};
    int L[3] = {XS, YS, ZS};
    int M[3] = {XWM, YWM, ZWM}; /* margin */
    int T[3] = {XTE, YTE, ZTE}; /* texture */

    for (c = 0; c < 3; ++c) {
        t = T[c] * (r[c] + L[c] / 2 + M[c]) / (L[c] + 2 * M[c]) - 0.5;
        lmbd[c] = t - (int)t;
        tc[c] = (int)t + 0.5;
    }
#define tex0(ix, iy, iz) (texsdf.fetch(tc[0] + ix, tc[1] + iy, tc[2] + iz))
    s000 = tex0(0, 0, 0), s001 = tex0(1, 0, 0), s010 = tex0(0, 1, 0);
    s011 = tex0(1, 1, 0), s100 = tex0(0, 0, 1), s101 = tex0(1, 0, 1);
    s110 = tex0(0, 1, 1), s111 = tex0(1, 1, 1);
#undef tex0

#define wavrg(A, B, p) A*(1-p) + p*B /* weighted average */
    s00x = wavrg(s000, s001, lmbd[0]);
    s01x = wavrg(s010, s011, lmbd[0]);
    s10x = wavrg(s100, s101, lmbd[0]);
    s11x = wavrg(s110, s111, lmbd[0]);

    s0yx = wavrg(s00x, s01x, lmbd[1]);

    s1yx = wavrg(s10x, s11x, lmbd[1]);
    szyx = wavrg(s0yx, s1yx, lmbd[2]);
#undef wavrg
    return szyx;
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

__global__ void fill(const tex3Dca texsdf, const Particle *const pp, const int n,
                     int *const key) {
    enum {X, Y, Z};
    int pid = threadIdx.x + blockDim.x * blockIdx.x;
    if (pid >= n) return;
    Particle p = pp[pid];
    float sdf0 = sdf(texsdf, p.r[X], p.r[Y], p.r[Z]);
    key[pid] = (int)(sdf0 >= 0) + (int)(sdf0 > 2);
}
}}} /* namespace */
