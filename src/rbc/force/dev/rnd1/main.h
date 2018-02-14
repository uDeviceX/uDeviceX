struct Rnd0 { real r; };

static __device__ void edg_rnd(Shape shape, int i0, real *rnd, int  j, /**/ Rnd0 *rnd0) {
    /* i0: edge index; j: vertex index */
    int i1;
    i1 = shape.anti[i0];
    if (i1 > i0) j = j - i0 + i1;
    rnd0->r = rnd[j];
}

static __device__ real  frnd0(real dt, RbcParams_v par, real rnd) {
    real f, g, T;
    g = par.gammaC; T = par.kBT;
    f  = sqrtf(2*g*T/dt)*rnd;
    return f;
}

static __device__ real3 frnd(real dt, RbcParams_v par, real3 r1, real3 r2, const Rnd0 rnd) { /* random force */
    real3 dr, f;
    real r, f0;
    diff(&r1, &r2, /**/ &dr);
    r = sqrtf(dot<real>(&dr, &dr));
    f0 = frnd0(dt, par, rnd.r);
    axpy(f0/r, &dr, /**/ &f);
    return f;
}
