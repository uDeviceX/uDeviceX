struct Rnd0 { real r; };

static __device__ void edg_rnd(Shape shape, int i0, real* rnd, int  j, /**/ Rnd0 *rnd0) {
    /* i0: edge index; j: vertex index */
    assert(i0  < RBCnv * RBCmd); assert(j < MAX_CELL_NUM * RBCnv * RBCmd);
    int i1;
    i1 = shape.anti[i0];
    if (i1 > i0) j = j - i0 + i1;
    assert(j < MAX_CELL_NUM * RBCnv * RBCmd);
    rnd0->r = rnd[j];
}

static __device__ real  frnd0(RbcParams_v par, real rnd) {
    real f, g, T, dt0;
    g = par.gammaC; T = par.kBT0; dt0 = par.dt0;
    f  = sqrtf(2*g*T/dt0)*rnd;
    return f;
}

static __device__ real3 frnd(RbcParams_v par, real3 r1, real3 r2, const Rnd0 rnd) { /* random force */
    real3 dr, f;
    real r, f0;
    diff(&r1, &r2, /**/ &dr);
    r = sqrtf(dot<real>(&dr, &dr));
    f0 = frnd0(par, rnd.r);
    axpy(f0/r, &dr, /**/ &f);
    return f;
}
