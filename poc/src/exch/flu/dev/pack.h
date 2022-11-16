/* copy particles from bulk array pp to a single fragment to be exchanged */
__global__ void collect_particles(int fid, int nc, const Particle *pp, const int *bss,
                                  const int *bcc, const int *fss, int cap,
                                  /**/ int *fii, Particle *fpp, int *fnn) {
    int cid, tid, src, dst, nsrc, nfloat2s;
    int i, lpid, dpid, spid, c;

    /* 16 workers (warpSize/2) per cell  */
    /* requirement: 32 threads per block */
    /* cid: work group id = cell id      */
    /* tid: worker id within the group   */


    cid = threadIdx.x / (warpSize / 2) + 2 * blockIdx.x;
    tid = threadIdx.x % (warpSize / 2);

    if (cid >= nc) return;

    src = bss[cid];
    dst = fss[cid];
    nsrc = min(bcc[cid], cap - dst);

    const float2 *srcpp = (const float2*) pp;
    float2 *dstpp = (float2*) fpp;

    nfloat2s = nsrc * 3;
    for (i = tid; i < nfloat2s; i += warpSize/2) {
        lpid = i / 3;
        c    = i % 3;
        dpid = dst + lpid;
        spid = src + lpid;
        dstpp[3 * dpid + c] = srcpp[3 * spid + c];
    }
    for (lpid = tid; lpid < nsrc; lpid += warpSize / 2) {
        dpid = dst + lpid;
        spid = src + lpid;
        fii[dpid] = spid;
    }
    if (cid + 1 == nc) fnn[fid] = dst;
}

/* copy colors from bulk array ii to a single fragment to be exchanged */
__global__ void collect_colors(int fid, int nc, const int *ii,
                               const int *bss, const int *bcc, const int *fss,
                               int cap, /**/ int *fii) {
    int cid, tid, src, dst, nsrc;
    int lpid, dpid, spid;

    /* 16 workers (warpSize/2) per cell  */
    /* requirement: 32 threads per block */
    /* cid: work group id = cell id      */
    /* tid: worker id within the group   */

    cid = threadIdx.x / (warpSize / 2) + 2 * blockIdx.x;
    tid = threadIdx.x % (warpSize / 2);

    if (cid >= nc) return;

    src = bss[cid];
    dst = fss[cid];
    nsrc = min(bcc[cid], cap - dst);

    for (lpid = tid; lpid < nsrc; lpid += warpSize / 2) {
        dpid = dst + lpid;
        spid = src + lpid;
        fii[dpid] = ii[spid];
    }
}

