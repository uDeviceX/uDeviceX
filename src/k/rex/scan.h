namespace k_rex {
__global__ void scan(const int *counts, const int *oldtotalcounts,
                     /**/ int *totalcounts, int *paddedstarts) {
    int tid, mycount, newcount, myscan, L;

    tid = threadIdx.x;
    mycount = 0;

    if (tid < 26) {
        mycount = counts[tid];

        if (mycount > ccapacities[tid]) failed = true;

        if (totalcounts && oldtotalcounts) {
            newcount = mycount + oldtotalcounts[tid];
            totalcounts[tid] = newcount;

            if (newcount > ccapacities[tid]) failed = true;
        }
    }

    if (paddedstarts) {
        myscan = mycount = 32 * ((mycount + 31) / 32);

        for (int L = 1; L < 32; L <<= 1)
        myscan += (tid >= L) * __shfl_up(myscan, L);

        if (tid < 27) paddedstarts[tid] = myscan - mycount;
    }
}
}
