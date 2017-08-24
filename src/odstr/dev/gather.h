namespace odstr { namespace sub { namespace dev {
/** gather_id is in dev/common.h */

__global__ void gather_pp(const float2  *pp_lo, const float2 *pp_re, int n, const uint *iidx,
                          /**/ float2  *pp, float4  *zip0, ushort4 *zip1) {
    /* pp_lo, pp_re, pp: local, remote and output particles */
    int warp, tid, base, pid;
    bool valid, remote;
    uint spid;
    float2 d0, d1, d2; /* data */
    int nsrc, src0, src1, start, destbase;
    float3 s0, s1;

    warp = threadIdx.x / warpSize;
    tid = threadIdx.x % warpSize;

    base = warpSize * warp + blockDim.x * blockIdx.x;
    pid = base + tid;

    valid = (pid < n);

    if (valid) spid = iidx[pid];

    if (valid) {
        remote = (spid >> 31) & 1;
        spid &= ~(1 << 31);
        if (remote) {
            d0 = __ldg(pp_re + 0 + 3 * spid);
            d1 = __ldg(pp_re + 1 + 3 * spid);
            d2 = __ldg(pp_re + 2 + 3 * spid);
        } else {
            d0 = pp_lo[0 + 3 * spid];
            d1 = pp_lo[1 + 3 * spid];
            d2 = pp_lo[2 + 3 * spid];
        }
    }
    nsrc = min(32, n - base);

    src0 = (32 * ((tid    ) & 0x1) + tid) >> 1;
    src1 = (32 * ((tid + 1) & 0x1) + tid) >> 1;
    start = tid % 2;
    destbase = 2 * base;

    s0 = make_float3(d0.x, d0.y, d1.x);
    s1 = make_float3(d1.y, d2.x, d2.y);

    xchg_aos4f(src0, src1, start, s0, s1);

    if (tid < 2 * nsrc)
        zip0[destbase + tid] = make_float4(s0.x, s0.y, s0.z, 0);

    if (tid + 32 < 2 * nsrc)
        zip0[destbase + tid + 32] = make_float4(s1.x, s1.y, s1.z, 0);

    if (tid < nsrc)
        zip1[base + tid] = make_ushort4(__float2half_rn(d0.x),
                                        __float2half_rn(d0.y),
                                        __float2half_rn(d1.x),
                                        0);
    k_write::AOS6f(pp + 3 * base, nsrc, d0, d1, d2);
}

}}} /* namespace */
