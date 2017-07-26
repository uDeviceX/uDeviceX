namespace k_rex {
__global__ void scanA(const int *counts, const int *oldtotalcounts,
                     /**/ int *totalcounts, int *starts) {
    int tid, cnt, newcount, scan, L;
    tid = threadIdx.x;
    cnt = 0;
    if (tid < 26) {
        cnt = counts[tid];
        if (cnt > g::ccapacities[tid]) g::failed = true;
        if (totalcounts && oldtotalcounts) {
            newcount = cnt + oldtotalcounts[tid];
            totalcounts[tid] = newcount;
            if (newcount > g::ccapacities[tid]) g::failed = true;
        }
    }

    if (starts) {
        scan = cnt = 32 * ((cnt + 31) / 32);
        for (L = 1; L < 32; L <<= 1)
            scan += (tid >= L) * __shfl_up(scan, L);
        if (tid < 27) starts[tid] = scan - cnt;
    }
}

__global__ void scanB(const int *count, /**/ int *start) {
    int tid, cnt, scan, L;
    tid = threadIdx.x;
    cnt = 0;
    if (tid < 26) {
        cnt = count[tid];
        if (cnt > g::ccapacities[tid]) g::failed = true;
    }
    if (start) {
        scan = cnt = 32 * ((cnt + 31) / 32);
        for (L = 1; L < 32; L <<= 1)
            scan += (tid >= L) * __shfl_up(scan, L);
        if (tid < 27) start[tid] = scan - cnt;
    }
}

}
