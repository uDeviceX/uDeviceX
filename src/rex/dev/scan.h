namespace dev {
__device__ void scan_pad(int cnt, int t, /**/ int *starts) {
    int L, scan;
    scan = cnt = 32 * ((cnt + 31) / 32);
    for (L = 1; L < 32; L <<= 1) scan += (t >= L) * __shfl_up(scan, L);
    if (t < 27) starts[t] = scan - cnt;
}

__global__ void scanA(const int *counts, const int *oldtcounts, /**/ int *tcounts, int *starts) {
    int t, cnt, newcount;
    t = threadIdx.x;
    cnt = 0;
    if (t < 26) {
        cnt = counts[t];
        if (tcounts && oldtcounts) {
            newcount = cnt + oldtcounts[t];
            tcounts[t] = newcount;
        }
    }
    if (starts) scan_pad(cnt, t, /**/ starts);
}

__global__ void scanB(const int *count, /**/ int *starts) {
    int t, cnt;
    t = threadIdx.x;
    cnt = 0;
    if (t < 26) cnt = count[t];
    if (starts) scan_pad(cnt, t, /**/ starts);
}

}
