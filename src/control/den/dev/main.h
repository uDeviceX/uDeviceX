static __device__ int warpReduceSum(int v) {
    for (int offset = warpSize>>1; offset > 0; offset >>= 1)
        v += __shfl_down(v, offset);
    return v;
}

__global__ void kill(int maxden, const int *starts, const int *counts, int n, const int *cids, /**/ int *ndead, int *kk) {
    int i, nd, cid, s, c, pid, j;
    nd = 0;
    c = maxden;
    i = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (i < n) {
        cid = cids[i];
        c = counts[cid];
        s = starts[cid];
    }

    nd = max(0, c - maxden);

    // TODO select randomly
    if (i < n) {
        for (j = 0; j < nd; ++j) {
            pid = s + j;
            kk[pid] = 1; // mark dead
        }
    }
        
    // count number of killed in the warp
    nd = warpReduceSum(nd);
    if (threadIdx.x % warpSize == 0)
        atomicAdd(ndead, nd);
}
