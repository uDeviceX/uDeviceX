namespace k_rex {
__global__ void scan(const int *counts, const int *oldtotalcounts,
                     /**/ int *totalcounts, int *paddedstarts) {
    int tid = threadIdx.x;

    int mycount = 0;

    if (tid < 26) {
        mycount = counts[tid];

        if (mycount > ccapacities[tid]) failed = true;

        if (totalcounts && oldtotalcounts) {
            int newcount = mycount + oldtotalcounts[tid];
            totalcounts[tid] = newcount;

            if (newcount > ccapacities[tid]) failed = true;
        }
    }

    if (paddedstarts) {
        int myscan = mycount = 32 * ((mycount + 31) / 32);

        for (int L = 1; L < 32; L <<= 1)
        myscan += (tid >= L) * __shfl_up(myscan, L);

        if (tid < 27) paddedstarts[tid] = myscan - mycount;
    }
}
}

