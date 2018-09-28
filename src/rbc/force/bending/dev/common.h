enum {SPRING_OK, SPRING_LONG};
static __device__ double sq(double x) { return x * x; }
static __device__ double wlc0(double r) { return (4*sq(r)-9*r+6)/(4*sq(r-1)); }
static __device__ double wlc(double lmax, double ks, double r) { return ks/lmax*wlc0(r/lmax); }
static __device__ double3 fspring(const RbcParams_v *par, double3 x21, double l0, /**/ int *pstatus) {
  #define wlc_r(r) (wlc(lmax, ks, r))
    double m, r, fwlc, fpow, lmax, ks, x0;
    *pstatus = SPRING_OK;
    ks = par->ks; m = par->mpow; x0 = par->x0;
    r = sqrtf(dot<double>(&x21, &x21));
    lmax = l0 / x0;
    if (r >= lmax) {
        *pstatus = SPRING_LONG;
        r = RBC_SPRING_CAP * lmax;
    }
    fwlc =   wlc_r(r); /* make fwlc + fpow = 0 for r = l0 */
    fpow = - wlc_r(l0) * powf(l0, m + 1) / powf(r, m + 1);
    scal(fwlc + fpow, /*io*/ &x21);
    return x21;
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
    double3 f, fv, fs;
    double3 x21, x32, x31;

    diff(&r2, &r1, /**/ &x21);
    diff(&r3, &r2, /**/ &x32);
    diff(&r3, &r1, /**/ &x31);

    fs = fspring(par, x21, si.l0, &spring_status);
#ifdef RBC_SPRING_FAIL
    if (spring_status != SPRING_OK) report_tri(r1, r2, r3);
#endif
    add(&fs, /*io*/ &f);
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
    double3 f;
    f.x = f.y = f.z = 0;
    return f;
}

static __device__ double  frnd0(double dt, const RbcParams_v *par, double rnd) {
    double f, g, T;
    g = par->gammaC; T = par->kBT;
    f  = sqrtf(2*g*T/dt)*rnd;
    return f;
}

static __device__ double3 frnd(double dt, const RbcParams_v *par, double3 r1, double3 r2, Rnd1Info rnd) {
    double3 dr;
    double r, f0;
    diff(&r1, &r2, /**/ &dr);
    r = sqrtf(dot<double>(&dr, &dr));
    f0 = frnd0(dt, par, rnd.r);
    scal(f0/r, /*io*/ &dr);
    return dr;
}
