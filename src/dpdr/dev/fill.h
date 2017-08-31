namespace dpdr { namespace sub { namespace dev {
__global__ void fill(const int27 cellpackstarts, const Particle *pp, const intp26 fragss,
                     const intp26 fragcc, const intp26 fragcum, const int26 fragcapacity,
                     /**/ intp26 fragindices, Particlep26 fragpp, int *required_bag_size) {
    int gid, fid, hci, tid, src, dst, nsrc, nfloat2s;
    int i, lpid, dpid, spid, c;

    /* 16 workers (warpSize/2) per cell  */
    /* requirement: 32 threads per block */
    /* gid: work group id                */
    /* tid: worker id within the group   */

    gid = (2 * threadIdx.x) / warpSize + 2 * blockIdx.x;
    tid = (2 * threadIdx.x) % warpSize ;

    if (gid >= cellpackstarts.d[26]) return;

    fid = k_common::fid(cellpackstarts.d, gid);
    hci = gid - cellpackstarts.d[fid];

    src = fragss.d[fid][hci];
    dst = fragcum.d[fid][hci];
    nsrc = min(fragcc.d[fid][hci], fragcapacity.d[fid] - dst);

    const float2 *srcpp = (const float2*) pp;
    float2 *dstpp = (float2*) fragpp.d[fid];

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
        fragindices.d[fid][dpid] = spid;
    }
    if (gid + 1 == cellpackstarts.d[fid + 1]) required_bag_size[fid] = dst;
}

__global__ void fill__ii(const int27 cellpackstarts, const int *ii,
                         const intp26 fragss, const intp26 fragcc, const intp26 fragcum,
                         const int26 fragcapacity, /**/ intp26 fragii) {
    int gid, fid, hci, tid, src, dst, nsrc;
    int lpid, dpid, spid;

    /* 16 workers (warpSize/2) per cell  */
    /* requirement: 32 threads per block */
    /* gid: work group id                */
    /* tid: worker id within the group   */

    gid = (threadIdx.x >> 4) + 2 * blockIdx.x;
    if (gid >= cellpackstarts.d[26]) return;

    fid = k_common::fid(cellpackstarts.d, gid);
    hci = gid - cellpackstarts.d[fid];

    tid = threadIdx.x & 0xf;
    src = fragss.d[fid][hci];
    dst = fragcum.d[fid][hci];
    nsrc = min(fragcc.d[fid][hci], fragcapacity.d[fid] - dst);

    for (lpid = tid; lpid < nsrc; lpid += warpSize / 2) {
        dpid = dst + lpid;
        spid = src + lpid;
        fragii.d[fid][dpid] = ii[spid];
    }
}

}}} /* namespace */
