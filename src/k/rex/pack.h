namespace k_rex {
__device__ void pack0(const float2 *pp, /**/ float2 *buf) {
    int warp, npack_padded, localbase, fid, packbase, npack, lane;
    float2 s0, s1, s2;
    int entry, pid, entry2;

    warp = threadIdx.x / warpSize;
    lane = threadIdx.x % warpSize;
    npack_padded = g::starts[26];
    for (localbase = 32 * (warp + 4 * blockIdx.x); localbase < npack_padded;
         localbase += gridDim.x * blockDim.x) {
        fid = k_common::fid(g::starts, localbase);
        packbase = localbase - g::starts[fid];
        npack = min(32, g::counts[fid] - packbase);        
        if (lane < npack) {
            entry = g::offsets[fid] + packbase + lane;
            pid = __ldg(g::scattered_indices[fid] + entry);

            entry2 = 3 * pid;

            s0 = __ldg(pp + entry2);
            s1 = __ldg(pp + entry2 + 1);
            s2 = __ldg(pp + entry2 + 2);

            s0.x -= ((fid + 2) % 3 - 1) * XS;
            s0.y -= ((fid / 3 + 2) % 3 - 1) * YS;
            s1.x -= ((fid / 9 + 2) % 3 - 1) * ZS;
        }
        k_write::AOS6f(buf + 3 * (g::tstarts[fid] + g::offsets[fid] + packbase), npack, s0, s1, s2);
    }
}

__global__ void pack(const float2 *pp, /**/ float2 *buf) {
    if (g::failed) return;
    pack0(pp, buf);
}

}
