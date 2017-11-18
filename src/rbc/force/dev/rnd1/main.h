struct Rnd0 { float r; };

static __device__ void edg_rnd(float *rnd, int i, /**/ Rnd0 *rnd0)  {  /* unpack random */
    rnd0->r = rnd[i];
}

static __device__ float3 frnd(float3, float3, const Rnd0 rnd) { /* random force */
    printf("rnd: %g\n", rnd.r);
    return make_float3(0, 0, 0);
}
