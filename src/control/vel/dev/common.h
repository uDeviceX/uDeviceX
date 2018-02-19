static __device__ float3 warpReduceSum(float3 val) {
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val.x += __shfl_down(val.x, offset);
        val.y += __shfl_down(val.y, offset);
        val.z += __shfl_down(val.z, offset);
    }
    return val;
}

static __device__ int warpReduceSum(int val) {
    for (int offset = warpSize/2; offset > 0; offset /= 2)
        val += __shfl_down(val, offset);
    return val;
}

__global__ void reduceByWarp(const float3 *gvel, const int *gnum, const uint ncells, /**/ float3 *vel, int *num) {
    assert(blockDim.x == 32);
    int i, c;
    float3 v = make_float3(0, 0, 0);
    int n = 0;
    i = threadIdx.x + blockIdx.x * blockDim.x;
    c = blockIdx.x; /* [c]hunk id */

    if (i < ncells) {
        v = gvel[i];
        n = gnum[i];
    }
    
    v = warpReduceSum(v);
    n = warpReduceSum(n);

    if ((threadIdx.x % warpSize) == 0) {
        vel[c] = v;
        num[c] = n;
    }
}
