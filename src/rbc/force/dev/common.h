static __device__ real3 fvolume(RbcParams_v par, real3 r2, real3 r3, real v0, real v) {
    real f0;
    real3 f, n;
    cross(&r3, &r2, /**/ &n);
    f0 = par.kv * (v - v0) / (6 * v0);
    axpy(f0, &n, /**/ &f);
    return f;
}

static __device__ real3 farea(RbcParams_v par, real3 x21, real3 x31, real3 x32,   real a0, real A0, real A) {
    real3 nn, nnx32, f;
    real a, f0, fa, fA;
    cross(&x21, &x31, /**/ &nn); /* normal */
    cross(&nn, &x32, /**/ &nnx32);
    a = 0.5 * sqrt(dot<real>(&nn, &nn));

    fA = - par.ka * (A - A0) / (4 * A0 * a);
    fa = - par.kd * (a - a0) / (4 * a0 * a);
    f0 = fA + fa;
    axpy(f0, &nnx32, /**/ &f);
    return f;
}

static __device__ void assert_r(real a, real m) {
    if (a < m) return;
    printf("a = %g >= max = %g\n", a, m);
    assert(0);
}
static __device__ real sq(real x) { return x * x; }
static __device__ real wlc0(real r) { return (4*sq(r)-9*r+6)/(4*sq(r-1)); }
static __device__ real wlc(real lmax, real ks, real r) {
    assert_r(r, lmax);
    return ks/lmax*wlc0(r/lmax);
}
static __device__ real3 fspring(RbcParams_v par, real3 x21, real l0) {
  #define wlc_r(r) (wlc(lmax, ks, r))
    real m;
    real r, fwlc, fpow, lmax, ks, x0;
    real3 f;
    ks = par.ks; m = par.mpow; x0 = par.x0;

    r = sqrtf(dot<real>(&x21, &x21));
    lmax = l0 / x0;
    fwlc =   wlc_r(r); /* make fwlc + fpow = 0 for r = l0 */
    fpow = - wlc_r(l0) * powf(l0, m + 1) / powf(r, m + 1);
    axpy(fwlc + fpow, &x21, /**/ &f);
    return f;
  #undef wlc_r
}

static __device__ real3 ftri0(RbcParams_v par, real3 r1, real3 r2, real3 r3,
                             real l0, real A0, real totArea, real totVolume,
                             real area, real volume) {
    real3 fv, fa, fs;
    real3 x21, x32, x31, f = make_real3(0, 0, 0);

    diff(&r2, &r1, /**/ &x21);
    diff(&r3, &r2, /**/ &x32);
    diff(&r3, &r1, /**/ &x31);

    fa = farea(par, x21, x31, x32,   A0, totArea, area);
    add(&fa, /**/ &f);

    fv = fvolume(par, r2, r3, totVolume, volume);
    add(&fv, /**/ &f);

    fs = fspring(par, x21,  l0);
    add(&fs, /**/ &f);

    return f;
}

static __device__ real3 fvisc(RbcParams_v par, real3 r1, real3 r2, real3 u1, real3 u2) {
    const real gammaC = par.gammaC, gammaT = par.gammaT;
    real3 du, dr, f = make_real3(0, 0, 0);
    diff(&u2, &u1, /**/ &du);
    diff(&r1, &r2, /**/ &dr);

    real fac = dot<real>(&du, &dr) / dot<real>(&dr, &dr);

    axpy(gammaT      , &du, /**/ &f);
    axpy(gammaC * fac, &dr, /**/ &f);

    return f;
}

/* forces from one dihedral */
template <int update>
__device__ real3 fdih(RbcParams_v par, real3 r1, real3 r2, real3 r3, real3 r4) {
    real overIksiI, overIdzeI, cosTheta, IsinThetaI2, sinTheta_1,
        beta, b11, b12, phi, sint0kb, cost0kb;
    real3 r12, r13, r34, r24, r41, ksi, dze, ksimdze;
    diff(&r1, &r2, /**/ &r12);
    diff(&r1, &r3, /**/ &r13);
    diff(&r3, &r4, /**/ &r34);
    diff(&r2, &r4, /**/ &r24);
    diff(&r4, &r1, /**/ &r41);

    cross(&r12, &r13, /**/ &ksi);
    cross(&r34, &r24, /**/ &dze);

    overIksiI = rsqrtf(dot<real>(&ksi, &ksi));
    overIdzeI = rsqrtf(dot<real>(&dze, &dze));

    cosTheta = dot<real>(&ksi, &dze) * overIksiI * overIdzeI;
    IsinThetaI2 = 1.0f - cosTheta * cosTheta;

    diff(&ksi, &dze, /**/ &ksimdze);

    sinTheta_1 = copysignf
        (rsqrtf(max(IsinThetaI2, 1.0e-6)),
         dot<real>(&ksimdze, &r41)); // ">" because the normals look inside

    phi = par.phi / 180.0 * M_PI;
    sint0kb = sin(phi) * par.kb;
    cost0kb = cos(phi) * par.kb;
    beta = cost0kb - cosTheta * sint0kb * sinTheta_1;

    b11 = -beta *  cosTheta * overIksiI * overIksiI;
    b12 =  beta * overIksiI * overIdzeI;

    if (update == 1) {
        real3 r32, f1, f;
        diff(&r3, &r2, /**/ &r32);
        cross(&ksi, &r32, /**/ &f);
        cross(&dze, &r32, /**/ &f1);
        scal(b11, /**/ &f);
        axpy(b12, &f1, /**/ &f);
        return f;
    }
    else if (update == 2) {
        real3 f, f1, f2, f3;
        real b22 = -beta * cosTheta * overIdzeI * overIdzeI;

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
        return make_real3(0, 0, 0);
}
