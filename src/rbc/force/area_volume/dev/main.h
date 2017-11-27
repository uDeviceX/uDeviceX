/* position - float2 union */
union Pos {
    float2 f2[2];
    struct { float3 r; float dummy; };
};

static __device__ Pos tex2Pos(const Texo<float2> texvert, const int id) {
    Pos r;
    r.f2[0] = fetch(texvert, 3 * id + 0);
    r.f2[1] = fetch(texvert, 3 * id + 1);
    return r;
}

static __device__ float2 warpReduceSum(float2 val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val.x += __shfl_down(val.x, offset);
        val.y += __shfl_down(val.y, offset);
    }
    return val;
}

static __device__ float area0(const float3 r0, const float3 r1, const float3 r2) {
    float3 x1, x2, n;
    diff(&r1, &r0, /**/ &x1);
    diff(&r2, &r0, /**/ &x2);
    cross(&x1, &x2, /**/ &n);
    return 0.5f * sqrtf(dot<float>(&n, &n));
}

static __device__ float volume0(float3 r0, float3 r1, float3 r2) {
    return
        0.1666666667f *
        ((r0.x*r1.y-r0.y*r1.x)*r2.z +
         (r0.z*r1.x-r0.x*r1.z)*r2.y +
         (r0.y*r1.z-r0.z*r1.y)*r2.x);
}

__global__ void dev(int nt, int nv, const Texo<float2> texvert, const Texo<int4> textri, float *totA_V) {
    float2 a_v = make_float2(0.0f, 0.0f);
    int cid = blockIdx.y;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nt;
         i += blockDim.x * gridDim.x) {
        int4 ids = fetch(textri, i);

        const Pos r0 = tex2Pos(texvert, ids.x + cid * nv);
        const Pos r1 = tex2Pos(texvert, ids.y + cid * nv);
        const Pos r2 = tex2Pos(texvert, ids.z + cid * nv);

        a_v.x += area0(r0.r, r1.r, r2.r);
        a_v.y += volume0(r0.r, r1.r, r2.r);
    }
    a_v = warpReduceSum(a_v);
    if ((threadIdx.x & (warpSize - 1)) == 0) {
        atomicAdd(&totA_V[2 * cid + 0], a_v.x);
        atomicAdd(&totA_V[2 * cid + 1], a_v.y);
    }
}
