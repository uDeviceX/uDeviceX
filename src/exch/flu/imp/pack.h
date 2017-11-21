void download_cell_starts(intp26 src, /**/ intp26 dst) {
    int i, nc;
    size_t sz;

    for(i = 0; i < NFRAGS; ++i) {
        nc = frag_ncell(i);
        sz = (nc + 1) * sizeof(int);
        d::MemcpyAsync(dst.d[i], src.d[i], sz, D2H);
    }
}

template <int NWARPS>
__global__ void scan(const int26 fragn, const intp26 fragcc, /**/ intp26 fragcum) {
    __shared__ int shdata[32];

    int tid, laneid, warpid, fid;
    int *count, *start, n;
    int lastval, sourcebase, sourceid, mycount, myscan;
    int L, val, gs;
    
    fid = blockIdx.x;
    count = fragcc.d[fid];
    start = fragcum.d[fid];
    n = fragn.d[fid];

    tid = threadIdx.x;
    laneid = threadIdx.x % warpSize;
    warpid = threadIdx.x / warpSize;

    lastval = 0;
    for (sourcebase = 0; sourcebase < n; sourcebase += 32 * NWARPS) {
        sourceid = sourcebase + tid;
        mycount = myscan = 0;
        if (sourceid < n) myscan = mycount = count[sourceid];
        if (tid == 0) myscan += lastval;

        for (L = 1; L < 32; L <<= 1) {
            i val = __shfl_up(myscan, L);
            if (laneid >= L) myscan += val;
        }

        if (laneid == 31) shdata[warpid] = myscan;
        __syncthreads();
        if (warpid == 0) {
            gs = 0;
            if (laneid < NWARPS) gs = shdata[tid];
            for ( L = 1; L < 32; L <<= 1) {
                val = __shfl_up(gs, L);
                if (laneid >= L) gs += val;
            }

            shdata[tid] = gs;
            lastval = __shfl(gs, 31);
        }
        __syncthreads();
        if (warpid) myscan += shdata[warpid - 1];
        __syncthreads();
        if (sourceid < n) start[sourceid] = myscan - mycount;
    }
}
