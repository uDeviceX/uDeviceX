struct Rnd0 { float r; };

static __device__ void edg_rnd(Shape shape, int i0, float* rnd, int  j, /**/ Rnd0 *rnd0) {
    /* i0: edge index; j: vertex index */
    assert(i0  < RBCmd * RBCnv); assert(j < MAX_CELL_NUM * RBCnv * RBCmd);
    int i1;
    i1 = shape.anti[i0];
    if (i1 > i0) j = j - i0 + i1;
    assert(j < MAX_CELL_NUM * RBCnv);
    rnd0->r = rnd[j];
}

static __device__ float  frnd0(float rnd) {
    float f, g, T;
    g = RBCgammaC; T = RBCkbT;
    f  = sqrtf(2*g*T/dt)*rnd;
    return f;
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
