namespace emap_dev {

static __device__ int warpexscan(int cnt, int t) {
    int L, scan;
    scan = cnt;
    for (L = 1; L < 32; L <<= 1) scan += (t >= L) * __shfl_up(scan, L);
    return scan - cnt;
}

__global__ void scan2d(int nfrags, const int *counts, const int *offsets, /**/ int *nextoffsets, int *starts) {
    int t, cnt, scan, nextoffset;
    t = threadIdx.x;
    cnt = 0;
    if (t < nfrags) {
        cnt = counts[t];
        nextoffset = cnt + offsets[t];
        nextoffsets[t] = nextoffset;
    }
    scan = warpexscan(cnt, t);
    if (t <= nfrags) starts[t] = scan;
}

} // dev
