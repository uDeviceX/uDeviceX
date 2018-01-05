static __device__ float fetch(Sdf_v *sdf, float i, float j, float k) {
    return Ttex3D(float, sdf->tex.t, i, j, k);
}
static __device__ int iround(float x) { return (x > 0.5) ? (x + 0.5) : (x - 0.5); }

static __device__ int small_diff(const float a[3], const float b[3]) {
    enum {X, Y, Z};
    const float eps = 1e-3;
    int cx, cy, cz;
    cx = -eps < a[X] - b[X] && a[X] - b[X] < eps;
    cy = -eps < a[Y] - b[Y] && a[Y] - b[Y] < eps;
    cz = -eps < a[Z] - b[Z] && a[Z] - b[Z] < eps;
    return cx && cy && cz;
}
static __device__ void convert(Sdf_v *sdf, const float a[3], /**/ float b[3]) {
    enum {X, Y, Z};
    int c;
    int L[3] = {XS, YS, ZS};
    int M[3] = {XWM, YWM, ZWM};
    int T[3] = {XTE, YTE, ZTE};
    for (c = 0; c < 3; ++c)
        b[c] = T[c] * (a[c] + L[c] / 2 + M[c]) / (L[c] + 2 * M[c]) - 0.5;

    float q[3];
    tform_convert_dev(&sdf->t, a, /**/ q);
    if (!small_diff(b, q)) {
        printf("convert failed: [%g %g %g] != [%g %g %g]\n", b[X], b[Y], b[Z], q[X], q[Y], q[Z]);
        assert(0);
    }
}
static __device__ void convert_floor(Sdf_v *sdf, const float a[3], /**/ int i[3]) {
    enum {X, Y, Z};
    float f[3];
    convert(sdf, a, /**/ f);
    i[X] = int(f[X]); i[Y] = int(f[Y]); i[Z] = int(f[Z]);
}
static __device__ void convert_round(Sdf_v *sdf, const float a[3], /**/ int i[3]) {
    enum {X, Y, Z};
    float f[3];
    convert(sdf, a, /**/ f);
    i[X] = iround(f[X]); i[Y] = iround(f[Y]); i[Z] = iround(f[Z]);
}
static __device__ void spacing(Sdf_v *sdf, float isp[3]) {
    int c;
    int L[3] = {XS, YS, ZS};
    int M[3] = {XWM, YWM, ZWM};
    int T[3] = {XTE, YTE, ZTE};
    for (c = 0; c < 3; ++c) isp[c] = T[c]/(2*M[c] + L[c]);
}

static __device__ float3 grad(Sdf_v *sdf, const float3 *pos) {
    int tc[3];
    float fcts[3], r[3] = {pos->x, pos->y, pos->z};
    float myval, gx, gy, gz;
    convert_floor(sdf, r, /**/ tc);
    spacing(sdf, /**/ fcts);

#define tex0(ix, iy, iz) (fetch(sdf, tc[0] + ix, tc[1] + iy, tc[2] + iz))
    myval = tex0(0, 0, 0);
    gx = fcts[0] * (tex0(1, 0, 0) - myval);
    gy = fcts[1] * (tex0(0, 1, 0) - myval);
    gz = fcts[2] * (tex0(0, 0, 1) - myval);
#undef tex0
    return make_float3(gx, gy, gz);
}

static __device__ float3 ugrad(Sdf_v *texsdf, const float3 *r) {
    float mag, eps;
    float3 g;
    eps = 1e-6;
    g = grad(texsdf, r);
    mag = sqrt(g.x*g.x + g.y*g.y + g.z*g.z);
    if (mag > eps) {
        g.x /= mag;
        g.y /= mag;
        g.z /= mag;
    }
    return g;
}

static __device__ float cheap_sdf(Sdf_v *sdf, float x, float y, float z)  {
    int tc[3];
    float r[3] = {x, y, z};
    convert_round(sdf, r, /**/ tc);
    return fetch(sdf, tc[0], tc[1], tc[2]);
}

static __device__ float sdf(Sdf_v *sdf0, float x, float y, float z) {
    int c;
    float s000, s001, s010, s100, s101, s011, s110, s111;
    float s00x, s01x, s10x, s11x;
    float s0yx, s1yx;
    float szyx;
    float tc0[3], tc[3], lmbd[3], r[3] = {x, y, z};
    convert(sdf0, r, /**/ tc0);
    for (c = 0; c < 3; ++c) {
        lmbd[c] = tc0[c] - (int)tc0[c];
        tc[c] = (int)tc0[c] + 0.5;
    }
#define tex0(ix, iy, iz) (fetch(sdf0, tc[0] + ix, tc[1] + iy, tc[2] + iz))
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
