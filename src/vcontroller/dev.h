namespace dev {

static __device__ bool valid(int3 L, int3 c) {
    return c.x < L.x && c.y < L.y && c.z < L.z;
}

static __device__ int get_cid(int3 L, int3 c) {
    return c.x + L.x * (c.y + L.y * c.z);
}

__global__ void sample(int3 L, const int *__restrict__ cellsstart, __restrict__ const float2 *pp, /**/ float3 *gridv) {
    const int3 c = make_int3(threadIdx.x + blockIdx.x * blockDim.x,
                             threadIdx.y + blockIdx.y * blockDim.y,
                             threadIdx.z + blockIdx.z * blockDim.z);

    if (valid(L, c)) {
        const int cid = get_cid(L, c);
        const float num = cellsstart[cid+1] - cellsstart[cid];

        for (int pid = cellsstart[cid]; pid < cellsstart[cid+1]; pid++) {
            float2 tmp1 = pp[3*pid + 1];
            float2 tmp2 = pp[3*pid + 2];
            gridv[cid].x += tmp1.y / num;
            gridv[cid].y += tmp2.x / num;
            gridv[cid].z += tmp2.y / num;
        }
    }
}

__device__ float3 warpReduceSum(float3 val) {
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val.x += __shfl_down(val.x, offset);
        val.y += __shfl_down(val.y, offset);
        val.z += __shfl_down(val.z, offset);
    }
    return val;
}

__global__ void reduceByWarp(const float3 * const __restrict__ vel, const uint ncells, /**/ float3 *res) {
    assert(blockDim.x == 32);
    int i, c;
    float3 v;
    i = threadIdx.x + blockIdx.x * blockDim.x;
    c = blockIdx.x; /* [c]hunk id */

    if (i >= ncells) return;

    v = vel[i];
    v = warpReduceSum(v);

    if ((threadIdx.x % warpSize) == 0)
        res[c] = v;
}

} // dev
