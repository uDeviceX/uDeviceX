static __device__ float3 fvolume(const RbcParams_v *par, float3 r2, float3 r3, float v0, float v) {
    float f0;
    float3 f;
    cross(&r3, &r2, /**/ &f);
    f0 = par->kv * (v - v0) / (6 * v0);
    scal(f0, &f);
    return f;
}

static __device__ float3 farea(const RbcParams_v *par, float3 x21, float3 x31, float3 x32,   float a0, float A0, float A) {
    float3 nn, nnx32, f;
    float a, f0, fa, fA;
    cross(&x21, &x31, /**/ &nn); /* normal */
    cross(&nn, &x32, /**/ &nnx32);
    a = 0.5 * sqrt(dot<float>(&nn, &nn));

    fA = - par->ka * (A - A0) / (4 * A0 * a);
    fa = - par->kd * (a - a0) / (4 * a0 * a);
    f0 = fA + fa;
    axpy(f0, &nnx32, /**/ &f);
    return f;
}

enum {SPRING_OK, SPRING_LONG};
static __device__ float sq(float x) { return x * x; }
static __device__ float wlc0(float r) { return (4*sq(r)-9*r+6)/(4*sq(r-1)); }
static __device__ float wlc(float lmax, float ks, float r) { return ks/lmax*wlc0(r/lmax); }
static __device__ float3 fspring(const RbcParams_v *par, float3 x21, float l0, /**/ int *pstatus) {
  #define wlc_r(r) (wlc(lmax, ks, r))
    float m, r, fwlc, fpow, lmax, ks, x0;
    float3 f;
    *pstatus = SPRING_OK;
    ks = par->ks; m = par->mpow; x0 = par->x0;
    r = sqrtf(dot<float>(&x21, &x21));
    lmax = l0 / x0;
    if (r >= lmax) {
        *pstatus = SPRING_LONG;
        return make_float3(0, 0, 0);
    }
    fwlc =   wlc_r(r); /* make fwlc + fpow = 0 for r = l0 */
    fpow = - wlc_r(l0) * powf(l0, m + 1) / powf(r, m + 1);
    axpy(fwlc + fpow, &x21, /**/ &f);
    return f;
  #undef wlc_r
}

static __device__ void report_tri(float3 r1, float3 r2, float3 r3) {
    printf("bad triangle: [%g %g %g] [%g %g %g] [%g %g %g]\n",
           r1.x, r1.y, r1.z,   r2.x, r2.y, r2.z,   r3.x, r3.y, r3.z);
    assert(0);
}
static __device__ float3 ftri(const RbcParams_v *par, float3 r1, float3 r2, float3 r3,
                             StressInfo si, float area, float volume) {
    int spring_status;
    float3 fv, fa, fs;
    float3 x21, x32, x31, f = make_float3(0, 0, 0);

    diff(&r2, &r1, /**/ &x21);
    diff(&r3, &r2, /**/ &x32);
    diff(&r3, &r1, /**/ &x31);

    fa = farea(par, x21, x31, x32, si.a0, par->totArea, area);
    add(&fa, /**/ &f);

    fv = fvolume(par, r2, r3, par->totVolume, volume);
    add(&fv, /**/ &f);

    fs = fspring(par, x21, si.l0, &spring_status);
    if (spring_status != SPRING_OK)
        report_tri(r1, r2, r3);
    
    add(&fs, /**/ &f);

    return f;
}

static __device__ float3 fvisc(const RbcParams_v *par, float3 r1, float3 r2, float3 u1, float3 u2) {
    const float gammaC = par->gammaC, gammaT = par->gammaT;
    float3 du, dr, f = make_float3(0, 0, 0);
    diff(&u2, &u1, /**/ &du);
    diff(&r1, &r2, /**/ &dr);

    float fac = dot<float>(&du, &dr) / dot<float>(&dr, &dr);

    axpy(gammaT      , &du, /**/ &f);
    axpy(gammaC * fac, &dr, /**/ &f);

    return f;
}

static __device__ float3 frnd(float, const RbcParams_v*, float3, float3, Rnd0Info) {
    return make_float3(0, 0, 0);
}

static __device__ float  frnd0(float dt, const RbcParams_v *par, float rnd) {
    float f, g, T;
    g = par->gammaC; T = par->kBT;
    f  = sqrtf(2*g*T/dt)*rnd;
    return f;
}

static __device__ float3 frnd(float dt, const RbcParams_v *par, float3 r1, float3 r2, Rnd1Info rnd) {
    float3 dr, f;
    float r, f0;
    diff(&r1, &r2, /**/ &dr);
    r = sqrtf(dot<float>(&dr, &dr));
    f0 = frnd0(dt, par, rnd.r);
    axpy(f0/r, &dr, /**/ &f);
    return f;
}

