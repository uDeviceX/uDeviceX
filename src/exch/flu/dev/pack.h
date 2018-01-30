/* copy particles from bulk array pp to fragments to be exchanged */
__global__ void collect_particles(const int27 fragstart, const Particle *pp, const intp26 bss,
                                  const intp26 bcc, const intp26 fss, const int26 cap,
                                  /**/ intp26 fii, Pap26 fpp, int *fnn) {
    int gid, fid, hci, tid, src, dst, nsrc, nfloat2s;
    int i, lpid, dpid, spid, c;

    /* 16 workers (warpSize/2) per cell  */
    /* requirement: 32 threads per block */
    /* gid: work group id                */
    /* tid: worker id within the group   */


    gid = threadIdx.x / (warpSize / 2) + 2 * blockIdx.x;
    tid = threadIdx.x % (warpSize / 2);

    if (gid >= fragstart.d[26]) return;

    fid = fragdev::frag_get_fid(fragstart.d, gid);
    hci = gid - fragstart.d[fid];

    src = bss.d[fid][hci];
    dst = fss.d[fid][hci];
    nsrc = min(bcc.d[fid][hci], cap.d[fid] - dst);

    const float2 *srcpp = (const float2*) pp;
    float2 *dstpp = (float2*) fpp.d[fid];

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
        fii.d[fid][dpid] = spid;
    }
    if (gid + 1 == fragstart.d[fid + 1]) fnn[fid] = dst;
}

/* copy colors from bulk array ii to fragments to be exchanged */
__global__ void collect_colors(const int27 fragstart, const int *ii,
                               const intp26 bss, const intp26 bcc, const intp26 fss,
                               const int26 cap, /**/ intp26 fii) {
    int gid, fid, hci, tid, src, dst, nsrc;
    int lpid, dpid, spid;

    /* 16 workers (warpSize/2) per cell  */
    /* requirement: 32 threads per block */
    /* gid: work group id                */
    /* tid: worker id within the group   */

    gid = threadIdx.x / (warpSize / 2) + 2 * blockIdx.x;
    tid = threadIdx.x % (warpSize / 2);

    if (gid >= fragstart.d[26]) return;

    fid = fragdev::frag_get_fid(fragstart.d, gid);
    hci = gid - fragstart.d[fid];

    src = bss.d[fid][hci];
    dst = fss.d[fid][hci];
    nsrc = min(bcc.d[fid][hci], cap.d[fid] - dst);

    for (lpid = tid; lpid < nsrc; lpid += warpSize / 2) {
        dpid = dst + lpid;
        spid = src + lpid;
        fii.d[fid][dpid] = ii[spid];
    }
}

