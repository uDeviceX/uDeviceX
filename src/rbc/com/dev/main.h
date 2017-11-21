// TODO put all warpReduce/scan in one common place
static __device__ float3 warpReduceSum(float3 v) {
    for (int offset = warpSize>>1; offset > 0; offset >>= 1) {
        v.x += __shfl_down(v.x, offset);
        v.y += __shfl_down(v.y, offset);
        v.z += __shfl_down(v.z, offset);
    }
    return v;
}

__global__ void reduce_position(int nv, const Particle *pp, /**/ float3 *rr) {
    int mid, i;
    float3 r;
    i   = threadIdx.x + blockIdx.x * blockDim.x;
    mid = blockIdx.y;

    r = make_float3(0, 0, 0);
    
    if (i < nv) {
        enum {X, Y, Z};
        const float *r0 = pp[mid * nv + i].r;
        r.x = r0[X];
        r.y = r0[Y];
        r.z = r0[Z];
    }

    r = warpReduceSum(r);

    if ((threadIdx.x % warpSize) == 0) {
        atomicAdd(&rr[mid].x, r.x);
        atomicAdd(&rr[mid].y, r.y);
        atomicAdd(&rr[mid].z, r.z);
    }
}
