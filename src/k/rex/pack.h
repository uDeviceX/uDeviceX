namespace k_rex {
__global__ void pack(const float2 *pp, int soluteid, /**/ float2 *buffer) {
    if (failed) return;

    int warpid = threadIdx.x >> 5;
    int npack_padded = cpaddedstarts[26];

    for (int localbase = 32 * (warpid + 4 * blockIdx.x); localbase < npack_padded;
         localbase += gridDim.x * blockDim.x) {
        int key9 = 9 * ((int)(localbase >= cpaddedstarts[9]) +
                        (int)(localbase >= cpaddedstarts[18]));

        int key3 = 3 * ((int)(localbase >= cpaddedstarts[key9 + 3]) +
                        (int)(localbase >= cpaddedstarts[key9 + 6]));

        int key1 = (int)(localbase >= cpaddedstarts[key9 + key3 + 1]) +
            (int)(localbase >= cpaddedstarts[key9 + key3 + 2]);

        int code = key9 + key3 + key1;
        int packbase = localbase - cpaddedstarts[code];

        int npack = min(32, ccounts[code] - packbase);

        int lane = threadIdx.x & 0x1f;

        float2 s0, s1, s2;

        if (lane < npack) {
            int entry = coffsets[code] + packbase + lane;
            int pid = __ldg(scattered_indices[code] + entry);

            int entry2 = 3 * pid;

            s0 = __ldg(pp + entry2);
            s1 = __ldg(pp + entry2 + 1);
            s2 = __ldg(pp + entry2 + 2);

            s0.x -= ((code + 2) % 3 - 1) * XS;
            s0.y -= ((code / 3 + 2) % 3 - 1) * YS;
            s1.x -= ((code / 9 + 2) % 3 - 1) * ZS;
        }
        k_common::write_AOS6f(buffer + 3 * (cbases[code] + coffsets[code] + packbase), npack,
                              s0, s1, s2);
    }
}
}
