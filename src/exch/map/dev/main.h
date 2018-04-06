namespace emap_dev {

static __device__ void warpexscan(int cnt, int t, /**/ int *starts) {
    int L, scan;
    scan = cnt;
    for (L = 1; L < 32; L <<= 1) scan += (t >= L) * __shfl_up(scan, L);
    if (t < 27) starts[t] = scan - cnt;
}

__global__ void scan2d(const int *counts, const int *offsets, /**/ int *nextoffsets, int *starts) {
    int t, cnt, nextoffset;
    t = threadIdx.x;
    cnt = 0;
    if (t < 26) {
        cnt = counts[t];
        nextoffset = cnt + offsets[t];
        nextoffsets[t] = nextoffset;
    }
    if (starts) warpexscan(cnt, t, /**/ starts);
}

} // dev
