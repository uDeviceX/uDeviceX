static __device__ double3 fvolume(const RbcParams_v *par, double3 r2, double3 r3, double v0, double v) {
    double f0;
    double3 f, n;
    cross(&r3, &r2, /**/ &n);
    f0 = par->kv * (v - v0) / (6 * v0);
    axpy(f0, &n, /**/ &f);
    return f;
}

static __device__ double3 farea(const RbcParams_v *par, double3 x21, double3 x31, double3 x32,   double a0, double A0, double A) {
    double3 nn, nnx32, f;
    double a, f0, fa, fA;
    cross(&x21, &x31, /**/ &nn); /* normal */
    cross(&nn, &x32, /**/ &nnx32);
    a = 0.5 * sqrt(dot<double>(&nn, &nn));

    fA = - par->ka * (A - A0) / (4 * A0 * a);
    fa = - par->kd * (a - a0) / (4 * a0 * a);
    f0 = fA + fa;
    axpy(f0, &nnx32, /**/ &f);
    return f;
}

enum {SPRING_OK, SPRING_LONG};
static __device__ double sq(double x) { return x * x; }
static __device__ double wlc0(double r) { return (4*sq(r)-9*r+6)/(4*sq(r-1)); }
static __device__ double wlc(double lmax, double ks, double r) { return ks/lmax*wlc0(r/lmax); }
static __device__ double3 fspring(const RbcParams_v *par, double3 x21, double l0, /**/ int *pstatus) {
  #define wlc_r(r) (wlc(lmax, ks, r))
    double m, r, fwlc, fpow, lmax, ks, x0;
    double3 f;
    *pstatus = SPRING_OK;
    ks = par->ks; m = par->mpow; x0 = par->x0;
    r = sqrtf(dot<double>(&x21, &x21));
    lmax = l0 / x0;
    if (r >= lmax) {
        *pstatus = SPRING_LONG;
        return make_double3(0, 0, 0);
    }
    fwlc =   wlc_r(r); /* make fwlc + fpow = 0 for r = l0 */
    fpow = - wlc_r(l0) * powf(l0, m + 1) / powf(r, m + 1);
    axpy(fwlc + fpow, &x21, /**/ &f);
    return f;
  #undef wlc_r
}

static __device__ void report_tri(double3 r1, double3 r2, double3 r3) {
    printf("bad triangle: [%g %g %g] [%g %g %g] [%g %g %g]\n",
           r1.x, r1.y, r1.z,   r2.x, r2.y, r2.z,   r3.x, r3.y, r3.z);
    assert(0);
}
static __device__ double3 ftri(const RbcParams_v *par, double3 r1, double3 r2, double3 r3,
                             StressInfo si, double area, double volume) {
    int spring_status;
    double3 fv, fa, fs;
    double3 x21, x32, x31, f = make_double3(0, 0, 0);

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

static __device__ double3 fvisc(const RbcParams_v *par, double3 r1, double3 r2, double3 u1, double3 u2) {
    const double gammaC = par->gammaC, gammaT = par->gammaT;
    double3 du, dr, f = make_double3(0, 0, 0);
    diff(&u2, &u1, /**/ &du);
    diff(&r1, &r2, /**/ &dr);

    double fac = dot<double>(&du, &dr) / dot<double>(&dr, &dr);

    axpy(gammaT      , &du, /**/ &f);
    axpy(gammaC * fac, &dr, /**/ &f);

    return f;
}

static __device__ double3 frnd(double, const RbcParams_v*, double3, double3, Rnd0Info) {
    return make_double3(0, 0, 0);
}

static __device__ double  frnd0(double dt, const RbcParams_v *par, double rnd) {
    double f, g, T;
    g = par->gammaC; T = par->kBT;
    f  = sqrtf(2*g*T/dt)*rnd;
    return f;
}

static __device__ double3 frnd(double dt, const RbcParams_v *par, double3 r1, double3 r2, Rnd1Info rnd) {
    double3 dr, f;
    double r, f0;
    diff(&r1, &r2, /**/ &dr);
    r = sqrtf(dot<double>(&dr, &dr));
    f0 = frnd0(dt, par, rnd.r);
    axpy(f0/r, &dr, /**/ &f);
    return f;
}

/* forces from one dihedral */
template <int update>
__device__ double3 fdih0(double phi, double kb,
                         double3 r1, double3 r2, double3 r3, double3 r4) {
    double overIksiI, overIdzeI, cosTheta, IsinThetaI2, sinTheta_1,
        beta, b11, b12, sint0kb, cost0kb;
    double3 r12, r13, r34, r24, r41, ksi, dze, ksimdze;
    diff(&r1, &r2, /**/ &r12);
    diff(&r1, &r3, /**/ &r13);
    diff(&r3, &r4, /**/ &r34);
    diff(&r2, &r4, /**/ &r24);
    diff(&r4, &r1, /**/ &r41);

    cross(&r12, &r13, /**/ &ksi);
    cross(&r34, &r24, /**/ &dze);

    overIksiI = rsqrtf(dot<double>(&ksi, &ksi));
    overIdzeI = rsqrtf(dot<double>(&dze, &dze));

    cosTheta = dot<double>(&ksi, &dze) * overIksiI * overIdzeI;
    IsinThetaI2 = 1.0f - cosTheta * cosTheta;

    diff(&ksi, &dze, /**/ &ksimdze);

    sinTheta_1 = copysignf
        (rsqrtf(max(IsinThetaI2, 1.0e-6)),
         dot<double>(&ksimdze, &r41)); // ">" because the normals look inside

    sint0kb = sin(phi) * kb;
    cost0kb = cos(phi) * kb;
    beta = cost0kb - cosTheta * sint0kb * sinTheta_1;

    b11 = -beta *  cosTheta * overIksiI * overIksiI;
    b12 =  beta * overIksiI * overIdzeI;

    if (update == 1) {
        double3 r32, f1, f;
        diff(&r3, &r2, /**/ &r32);
        cross(&ksi, &r32, /**/ &f);
        cross(&dze, &r32, /**/ &f1);
        scal(b11, /**/ &f);
        axpy(b12, &f1, /**/ &f);
        return f;
    }
    else if (update == 2) {
        double3 f, f1, f2, f3;
        double b22 = -beta * cosTheta * overIdzeI * overIdzeI;

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
        return make_double3(0, 0, 0);
}

static __device__ double3 fdih_a(double phi, double kb,
                          double3 a, double3 b, double3 c, double3 d) {
    return fdih0<1>(phi, kb, a, b, c, d);
}

static __device__ double3 fdih_b(double phi, double kb,
                          double3 a, double3 b, double3 c, double3 d) {
    return fdih0<2>(phi, kb, a, b, c, d);
}
