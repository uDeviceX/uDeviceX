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

#ifdef RBC_SPRING_FAIL
    if (spring_status != SPRING_OK) report_tri(r1, r2, r3);
#endif
    add(&fs, /*io*/ &f);
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
