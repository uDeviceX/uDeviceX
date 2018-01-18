namespace dev {

static __device__ void warpexscan(int cnt, int t, /**/ int *starts) {
    int L, scan;
    scan = cnt;
    for (L = 1; L < 32; L <<= 1) scan += (t >= L) * __shfl_up(scan, L);
    if (t < 27) starts[t] = scan - cnt;
}

__global__ void scan2d(const int *counts, const int *oldtcounts, /**/ int *tcounts, int *starts) {
    int t, cnt, newcnt;
    t = threadIdx.x;
    cnt = 0;
    if (t < 26) {
        cnt = counts[t];
        newcnt = cnt + oldtcounts[t];
        tcounts[t] = newcnt;
    }
    if (starts) warpexscan(cnt, t, /**/ starts);
}

} // dev
