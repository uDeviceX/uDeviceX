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
