struct Rnd0 { float r; };

static __device__ void edg_rnd(float *rnd, int i, /**/ Rnd0 *rnd0)  {
    rnd0->r = rnd[i];
}

static __device__ float3 frnd(float3, float3, const Rnd0 rnd) {
    //    printf("rnd: %g\n", rnd);
    return make_float3(0, 0, 0);
}
