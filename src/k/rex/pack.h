namespace k_rex {
__global__ void pack(const float2 *pp, /**/ float2 *buffer) {
    if (g::failed) return;

    int warpid = threadIdx.x >> 5;
    int npack_padded = g::cpaddedstarts[26];

    for (int localbase = 32 * (warpid + 4 * blockIdx.x); localbase < npack_padded;
         localbase += gridDim.x * blockDim.x) {
        int code = k_common::fid(g::cpaddedstarts, localbase);
        int packbase = localbase - g::cpaddedstarts[code];

        int npack = min(32, g::ccounts[code] - packbase);

        int lane = threadIdx.x & 0x1f;

        float2 s0, s1, s2;

        if (lane < npack) {
            int entry = g::coffsets[code] + packbase + lane;
            int pid = __ldg(g::scattered_indices[code] + entry);

            int entry2 = 3 * pid;

            s0 = __ldg(pp + entry2);
            s1 = __ldg(pp + entry2 + 1);
            s2 = __ldg(pp + entry2 + 2);

            s0.x -= ((code + 2) % 3 - 1) * XS;
            s0.y -= ((code / 3 + 2) % 3 - 1) * YS;
            s1.x -= ((code / 9 + 2) % 3 - 1) * ZS;
        }
        k_write::AOS6f(buffer + 3 * (g::cbases[code] + g::coffsets[code] + packbase), npack,
                       s0, s1, s2);
    }
}
}
