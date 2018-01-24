static __device__ float3 fvolume(RbcParams_v par, float3 r2, float3 r3, float v) {
    float v0, f0;
    float3 f, n;
    v0 = RBCtotVolume;

    cross(&r3, &r2, /**/ &n);
    f0 = par.kv * (v - v0) / (6 * v0);
    axpy(f0, &n, /**/ &f);
    return f;
}

static __device__ float3 farea(RbcParams_v par, float3 x21, float3 x31, float3 x32,   float a0, float A0, float A) {
    float3 nn, nnx32, f;
    float a, f0, fa, fA;
    
    cross(&x21, &x31, /**/ &nn); /* normal */
    cross(&nn, &x32, /**/ &nnx32);
    a = 0.5 * sqrtf(dot<float>(&nn, &nn));

    fA = - par.ka * (A - A0) / (4 * A0 * a);
    fa = - par.kd * (a - a0) / (4 * a0 * a);
    f0 = fA + fa;
    axpy(f0, &nnx32, /**/ &f);
    return f;
}

#define real double
static __device__ real sq(real x) { return x * x; }
static __device__ real wlc0(real r) { return (4*sq(r)-9*r+6)/(4*sq(r-1)); }
static __device__ real wlc(real kbT, real lmax, real p, real r) {
    return kbT/(lmax*p)*wlc0(r/lmax);
}
static __device__ float3 fspring(RbcParams_v par, float3 x21, float l0) {
  #define wlc_r(r) (wlc(kbT, lmax, p, r))
    float m;
    float r, fwlc, fpow, lmax, kbT, p, ks, x0;
    float3 f;
    kbT = par.kBT0; p = par.p; ks = par.ks; m = par.mpow; x0 = par.x0;
    assert( ks>=0.95*kbT/p && ks<=1.05*kbT/p ); //TEST FAIL IF ENABLED

    r = sqrtf(dot<float>(&x21, &x21));
    lmax = l0 / x0;
    fwlc =   wlc_r(r); /* make fwlc + fpow = 0 for r = l0 */
    fpow = - wlc_r(l0) * powf(l0, m + 1) / powf(r, m + 1);
    axpy(fwlc + fpow, &x21, /**/ &f);
    return f;
  #undef wlc_r
}

static __device__ float3 tri0(RbcParams_v par, float3 r1, float3 r2, float3 r3,
                              float l0, float A0, float totArea,
                              float area, float volume) {
    float3 fv, fa, fs;
    float3 x21, x32, x31, f = make_float3(0, 0, 0);

    diff(&r2, &r1, /**/ &x21);
    diff(&r3, &r2, /**/ &x32);
    diff(&r3, &r1, /**/ &x31);

    fa = farea(par, x21, x31, x32,   A0, totArea, area);
    add(&fa, /**/ &f);

    fv = fvolume(par, r2, r3, volume);
    add(&fv, /**/ &f);

    fs = fspring(par, x21,  l0);
    add(&fs, /**/ &f);

    return f;
}

static __device__ float3 visc(RbcParams_v par, float3 r1, float3 r2, float3 u1, float3 u2) {
    const float gammaC = par.gammaC, gammaT = par.gammaT;
    float3 du, dr, f = make_float3(0, 0, 0);
    diff(&u2, &u1, /**/ &du);
    diff(&r1, &r2, /**/ &dr);

    const float fac = dot<float>(&du, &dr) / dot<float>(&dr, &dr);

    axpy(gammaT      , &du, /**/ &f);
    axpy(gammaC * fac, &dr, /**/ &f);

    return f;
}

/* forces from one dihedral */
template <int update> __device__ float3 dih0(RbcParams_v par, float3 r1, float3 r2, float3 r3, float3 r4) {
    float overIksiI, overIdzeI, cosTheta, IsinThetaI2, sinTheta_1,
        beta, b11, b12, phi, sint0kb, cost0kb;

    float3 r12, r13, r34, r24, r41, ksi, dze, ksimdze;
    diff(&r1, &r2, /**/ &r12);
    diff(&r1, &r3, /**/ &r13);
    diff(&r3, &r4, /**/ &r34);
    diff(&r2, &r4, /**/ &r24);
    diff(&r4, &r1, /**/ &r41);

    cross(&r12, &r13, /**/ &ksi);
    cross(&r34, &r24, /**/ &dze);

    overIksiI = rsqrtf(dot<float>(&ksi, &ksi));
    overIdzeI = rsqrtf(dot<float>(&dze, &dze));

    cosTheta = dot<float>(&ksi, &dze) * overIksiI * overIdzeI;
    IsinThetaI2 = 1.0f - cosTheta * cosTheta;

    diff(&ksi, &dze, /**/ &ksimdze);

    sinTheta_1 = copysignf
        (rsqrtf(max(IsinThetaI2, 1.0e-6f)),
         dot<float>(&ksimdze, &r41)); // ">" because the normals look inside

    phi = par.phi / 180.0 * M_PI;
    sint0kb = sin(phi) * par.kb;
    cost0kb = cos(phi) * par.kb;
    beta = cost0kb - cosTheta * sint0kb * sinTheta_1;

    b11 = -beta *  cosTheta * overIksiI * overIksiI;
    b12 =  beta * overIksiI * overIdzeI;

    if (update == 1) {
        float3 r32, f1, f;
        diff(&r3, &r2, /**/ &r32);
        cross(&ksi, &r32, /**/ &f);
        cross(&dze, &r32, /**/ &f1);
        scal(b11, /**/ &f);
        axpy(b12, &f1, /**/ &f);
        return f;
    }
    else if (update == 2) {
        float3 f, f1, f2, f3;
        float b22 = -beta * cosTheta * overIdzeI * overIdzeI;

        cross(&ksi, &r13, /**/ &f);
        cross(&ksi, &r34, /**/ &f1);
        cross(&dze, &r13, /**/ &f2);
        cross(&dze, &r34, /**/ &f3);

        scal(b11, /**/ &f);
        add(&f2, /**/ &f1);
        axpy(b12, &f1, /**/ &f);
        axpy(b22, &f3, /**/ &f);

        return f;
    }
    else
        return make_float3(0, 0, 0);
}
