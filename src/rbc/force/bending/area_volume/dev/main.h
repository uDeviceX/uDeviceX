static __device__ float3 pp2r(const Particle *pp, const int id) {
    enum {X, Y, Z};
    float3 r;
    r.x = pp[id].r[X];
    r.y = pp[id].r[Y];
    r.z = pp[id].r[Z];
    return r;
}

static __device__ float area(const float3 r0, const float3 r1, const float3 r2) {
    float3 x1, x2, n;
    diff(&r1, &r0, /**/ &x1);
    diff(&r2, &r0, /**/ &x2);
    cross(&x1, &x2, /**/ &n);
    return 0.5f * sqrtf(dot<float>(&n, &n));
}

static __device__ float volume(float3 r0, float3 r1, float3 r2) {
    return
        0.1666666667f *
        ((r0.x*r1.y-r0.y*r1.x)*r2.z +
         (r0.z*r1.x-r0.x*r1.z)*r2.y +
         (r0.y*r1.z-r0.z*r1.y)*r2.x);
}

__global__ void main(int nt, int nv, const Particle *pp, const int4 *tri, float *totA_V) {
    float2 a_v = make_float2(0.0f, 0.0f);
    int i, cid = blockIdx.y;
    float3 r0, r1, r2;
    int4 ids;

    for (i = blockIdx.x * blockDim.x + threadIdx.x; i < nt; i += blockDim.x * gridDim.x) {
        ids = tri[i];
        r0 = pp2r(pp, ids.x + cid * nv);
        r1 = pp2r(pp, ids.y + cid * nv);
        r2 = pp2r(pp, ids.z + cid * nv);

        a_v.x += area  (r0, r1, r2);
        a_v.y += volume(r0, r1, r2);
    }
    a_v = warpReduceSum(a_v);
    if ((threadIdx.x & (warpSize - 1)) == 0) {
        atomicAdd(&totA_V[2 * cid + 0], a_v.x);
        atomicAdd(&totA_V[2 * cid + 1], a_v.y);
    }
}
