struct Rnd0 { float r; };

static __device__ void edg_rnd(float *rnd, int i, /**/ Rnd0 *rnd0)  {
    /* i: edge index */
    assert(i < MAX_CELL_NUM * RBCnv);
    rnd0->r = rnd[i];
}

static __device__ float  frnd0(float rnd) {
    float f0, g, T;
    g = RBCgammaC; T = RBCkbT;
    f0  = sqrtf(2*g*T/dt)*rnd;
    return f0;
}

static __device__ float3 frnd(float3 r1, float3 r2, const Rnd0 rnd) { /* random force */
    float3 dr, f;
    float r, f0;
    diff(&r1, &r2, /**/ &dr);
    r = sqrtf(dot<float>(&dr, &dr));
    f0 = frnd0(rnd.r);
    axpy(f0/r, &dr, /**/ &f);
    return f;
}
