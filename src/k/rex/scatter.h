namespace k_rex {
__global__ void scatter(const float2 *pp, const int n, /**/ int *counts) {
    int warpid = threadIdx.x / warpSize;
    int lane   = threadIdx.x % warpSize;
    int base   = 32 * (warpid + 4 * blockIdx.x);
    int nsrc   = min(32, n - base);
    float2 s0, s1, s2;
    k_read::AOS6f(pp + 3 * base, nsrc, s0, s1, s2);

    int pid = base + lane;
    if (lane >= nsrc) return;
    enum {
        HXSIZE = XS / 2,
        HYSIZE = YS / 2,
        HZSIZE = ZS / 2
    };
    int halocode[3] = {
        -1 + (int)(s0.x >= -HXSIZE + 1) + (int)(s0.x >= HXSIZE - 1),
        -1 + (int)(s0.y >= -HYSIZE + 1) + (int)(s0.y >= HYSIZE - 1),
        -1 + (int)(s1.x >= -HZSIZE + 1) + (int)(s1.x >= HZSIZE - 1)};
    if (halocode[0] == 0 && halocode[1] == 0 && halocode[2] == 0) return;
    // faces
    for (int d = 0; d < 3; ++d)
        if (halocode[d]) {
            int xterm = (halocode[0] * (d == 0) + 2) % 3;
            int yterm = (halocode[1] * (d == 1) + 2) % 3;
            int zterm = (halocode[2] * (d == 2) + 2) % 3;

            int fid = xterm + 3 * (yterm + 3 * zterm);
            int myid = g::offsets[fid] + atomicAdd(counts + fid, 1);

            if (myid < g::capacities[fid]) g::scattered_indices[fid][myid] = pid;
        }
    // edges
    for (int d = 0; d < 3; ++d)
        if (halocode[(d + 1) % 3] && halocode[(d + 2) % 3]) {
            int xterm = (halocode[0] * (d != 0) + 2) % 3;
            int yterm = (halocode[1] * (d != 1) + 2) % 3;
            int zterm = (halocode[2] * (d != 2) + 2) % 3;

            int fid = xterm + 3 * (yterm + 3 * zterm);
            int myid = g::offsets[fid] + atomicAdd(counts + fid, 1);

            if (myid < g::capacities[fid]) g::scattered_indices[fid][myid] = pid;
        }
    // one corner
    if (halocode[0] && halocode[1] && halocode[2]) {
        int xterm = (halocode[0] + 2) % 3;
        int yterm = (halocode[1] + 2) % 3;
        int zterm = (halocode[2] + 2) % 3;

        int fid = xterm + 3 * (yterm + 3 * zterm);
        int myid = g::offsets[fid] + atomicAdd(counts + fid, 1);

        if (myid < g::capacities[fid]) g::scattered_indices[fid][myid] = pid;
    }
}
}
