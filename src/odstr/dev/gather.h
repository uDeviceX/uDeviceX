namespace odstr { namespace sub { namespace dev {
/** gather_id is in dev/common.h */

struct FLo { /* [F]rom [lo]cation in memory */
    const float2 *lo, *re;
};

struct TLo { /* [T]o [lo]cation */
    float2 *pp;
    float4  *zip0;
    ushort4 *zip1;
};

struct Da { /* Data */
    float2 d0, d1, d2;
};

__device__ void ini_FLo(const float2 *lo, const float2 *re, /**/ FLo *l) {
    l->lo = lo; l->re = re;
}

__device__ void Flo2D(Flo *l, /**/ Da *d) {
    
}

__global__ void gather_pp(const float2  *pp_lo, const float2 *pp_re, int n, const uint *iidx,
                          /**/ float2  *pp, float4  *zip0, ushort4 *zip1) {
    /* pp_lo, pp_re, pp: local, remote and output particles */
    int dw, ws, pid;
    bool remote;
    uint spid;
    float2 d0, d1, d2; /* data */
    int nsrc, src0, src1, start, destbase;
    float3 s0, s1;

    warpco(&ws, &dw);
    
    pid = ws + dw;
    if (pid < n) {
        spid = iidx[pid];
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
    nsrc = min(32, n - ws);

    src0 = (32 * ((dw    ) & 0x1) + dw) >> 1;
    src1 = (32 * ((dw + 1) & 0x1) + dw) >> 1;
    start = dw % 2;
    destbase = 2 * ws;

    s0 = make_float3(d0.x, d0.y, d1.x);
    s1 = make_float3(d1.y, d2.x, d2.y);

    xchg_aos4f(src0, src1, start, s0, s1);

    if (dw < 2 * nsrc)
        zip0[destbase + dw] = make_float4(s0.x, s0.y, s0.z, 0);

    if (dw + 32 < 2 * nsrc)
        zip0[destbase + dw + 32] = make_float4(s1.x, s1.y, s1.z, 0);

    if (dw < nsrc)
        zip1[ws + dw] = make_ushort4(__float2half_rn(d0.x),
                                        __float2half_rn(d0.y),
                                        __float2half_rn(d1.x),
                                        0);
    k_write::AOS6f(pp + 3 * ws, nsrc, d0, d1, d2);
}

}}} /* namespace */
